/** Фасад движка: настройки, состояние GPU, доктор, очередь и все GPU-исполнители. */
import { createHash } from 'node:crypto'
import { writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { readdir, stat } from 'node:fs/promises'
import { basename, dirname, isAbsolute, join, resolve as resolvePath } from 'node:path'
import type {
  CallRecord,
  DeviceRef,
  GpuStatus,
  DoctorReport,
  FfmpegCaps,
  JobContext,
  JobOutput,
  JobRecord,
  LeaseRecord,
  ServerState,
  Settings,
  Who,
} from './types.js'
import { SettingsStore, gpuEnv } from './settings.js'
import { ensureDirs, files } from './paths.js'
import { detectGpu } from './gpu.js'
import { Doctor } from './doctor.js'
import type { PythonProbe } from './doctor.js'
import { JobManager, describeDevice } from './jobs.js'
import {
  applyProgressLine,
  buildFfmpegPlan,
  ensureOutputDir,
  ensureParent,
  probeMedia,
  type FfmpegParams,
  type ProbeInfo,
  type ProgressState,
} from './ffmpeg.js'
import { pythonProbe, pythonScript, runPythonWorker } from './python.js'
import { DeepseekOcrError, runDeepseekOcr, type DeepseekOcrResult } from './deepseek-ocr.js'
import { RustFaceWorker, rustFaceAvailable } from './face-rust.js'
import { run, stream } from './proc.js'

export const VERSION = '1.2.0'

/** Метаданные личности вызывающего из MCP-инструментов или HTTP-запросов. */
export interface SubmitMeta {
  tool: string
  argsHash: string
  label: string | null
  clientName: string | null
  who?: Who
}

export interface HistoryQuery {
  limit?: number
  kind?: string
  sessionId?: string
  operation?: string
}

type Logger = { info(message: string): void; warn(message: string): void }

/** Стабильная сериализация для сопоставления подсказок плагина и сервера. */
export function stableStringify(value: unknown): string {
  const seen = new WeakSet<object>()
  const walk = (input: unknown): unknown => {
    if (input === null || typeof input !== 'object') return input
    if (seen.has(input as object)) return null
    seen.add(input as object)
    if (Array.isArray(input)) return input.map(walk)
    const record = input as Record<string, unknown>
    const out: Record<string, unknown> = {}
    for (const key of Object.keys(record).sort()) out[key] = walk(record[key])
    return out
  }
  return JSON.stringify(walk(value ?? null))
}

function hashArgs(value: unknown): string {
  return createHash('sha256').update(stableStringify(value)).digest('hex').slice(0, 16)
}

function errWithCode(message: string, code: string, details?: unknown): Error & { code: string; details?: unknown } {
  return Object.assign(new Error(message), { code, details })
}

function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function resolveModel(settings: Settings, model: string): string {
  if (isAbsolute(model)) return model
  const inModels = join(settings.modelsDir, model)
  if (existsSync(inModels)) return inModels
  return resolvePath(model)
}

export class Engine {
  readonly settingsStore = new SettingsStore()
  readonly jobs: JobManager
  readonly doctor: Doctor
  readonly startedAt = Date.now()
  private gpuCache: GpuStatus = {
    available: false,
    passthrough: false,
    gpus: [],
    processes: [],
    error: 'not checked yet',
    hints: [],
    checkedAt: 0,
    source: 'none',
  }
  private gpuPromise: Promise<void> | null = null
  private refreshTimer: ReturnType<typeof setInterval> | null = null
  private stopSweeper: (() => void) | null = null

  /** Потоковая запись OCR-результата, отдаётся наружу в тестах. */
  static ocrFileWriter = writeFile

  /** Резолвер ключа DeepSeek из credential-store DSH; задаёт host-плагин. */
  deepseekKeyResolver: (() => Promise<string>) | null = null

  /** Постоянный Rust-воркер лиц: создаётся один раз и переживает задания. */
  private faceWorker: RustFaceWorker | null = null

  /** Ключ DeepSeek: сначала credential-store, затем переменная окружения. */
  async deepseekKey(settings: Settings): Promise<string> {
    try {
      if (this.deepseekKeyResolver) {
        const value = (await this.deepseekKeyResolver()).trim()
        if (value) return value
      }
    } catch {
      /* сервис credentials недоступен — пробуем окружение */
    }
    const name = (settings.deepseekApiKeyEnv || 'DEEPSEEK_API_KEY').trim()
    return (process.env[name] ?? '').trim()
  }

  constructor(private readonly logger: Logger = { info() {}, warn() {} }) {
    this.jobs = new JobManager({
      historyLimit: this.settingsStore.get().historyLimit,
      jobHistoryLimit: this.settingsStore.get().jobHistoryLimit,
      gpuConcurrency: this.settingsStore.get().gpuConcurrency,
      cpuConcurrency: this.settingsStore.get().cpuConcurrency,
      gpuAvailable: () => this.gpuCache.available,
      logger,
    })
    this.doctor = new Doctor({
      settings: () => this.settingsStore.get(),
      gpu: async (refresh) => await this.gpu(refresh),
      pythonProbe: async () => await this.probePython(),
    })
  }

  async boot(): Promise<void> {
    await ensureDirs()
    await this.jobs.load()
    await this.refreshGpu()
    this.refreshTimer = setInterval(() => {
      void this.refreshGpu()
    }, 20_000)
    this.refreshTimer.unref?.()
    this.stopSweeper = this.jobs.startSweeper()
    this.logger.info(`gputool v${VERSION} ready; ${this.gpuCache.available ? `${this.gpuCache.gpus.length} GPU` : 'GPU unavailable'}`)
  }

  async dispose(): Promise<void> {
    if (this.faceWorker) {
      await this.faceWorker.dispose().catch(() => undefined)
      this.faceWorker = null
    }
    if (this.refreshTimer) clearInterval(this.refreshTimer)
    this.refreshTimer = null
    this.stopSweeper?.()
    this.stopSweeper = null
  }

  async refreshGpu(): Promise<void> {
    if (this.gpuPromise) return await this.gpuPromise
    this.gpuPromise = (async () => {
      try {
        this.gpuCache = await detectGpu()
      } catch (error) {
        this.gpuCache = {
          ...this.gpuCache,
          available: false,
          error: error instanceof Error ? error.message : String(error),
          checkedAt: Date.now(),
        }
      } finally {
        this.gpuPromise = null
      }
    })()
    await this.gpuPromise
  }

  async gpu(refresh = false): Promise<typeof this.gpuCache> {
    if (refresh || Date.now() - this.gpuCache.checkedAt > 15_000) {
      await this.refreshGpu()
    }
    return this.gpuCache
  }

  async probePython(): Promise<PythonProbe | null> {
    return await pythonProbe(this.settingsStore.get())
  }

  async caps(gpuBinary: boolean): Promise<FfmpegCaps> {
    const settings = this.settingsStore.get()
    return await this.doctor.caps(gpuBinary ? settings.ffmpegGpuPath : settings.ffmpegPath)
  }

  settingsView(): Omit<Settings, 'apiToken'> {
    return this.settingsStore.view()
  }

  async updateSettings(patch: Partial<Settings>): Promise<Omit<Settings, 'apiToken'>> {
    const next = await this.settingsStore.update(patch)
    this.jobs.setLimits(next.historyLimit, next.jobHistoryLimit, next.gpuConcurrency, next.cpuConcurrency)
    this.doctor.invalidate()
    return this.settingsStore.view()
  }

  async doctorReport(refresh = false): Promise<DoctorReport> {
    return await this.doctor.report(refresh)
  }

  async state(): Promise<ServerState> {
    const [gpu, doctor] = await Promise.all([this.gpu(), this.doctorReport(false)])
    return {
      version: VERSION,
      uptimeSec: Math.round((Date.now() - this.startedAt) / 1000),
      apiBaseUrl: '',
      mcpCommand: this.mcpCommand(),
      gpu,
      doctor,
      jobs: this.jobs.list(50),
      queue: this.jobs.queueInfo(),
      leases: this.jobs.leasesList(),
      calls: this.jobs.callsList(30),
      settings: this.settingsView(),
    }
  }

  mcpCommand(): string {
    const self = process.argv[1] ?? ''
    const server = self.includes('mcp') ? self : join(dirname(self), 'mcp', 'index.js')
    return `node ${server} --stdio`
  }

  private resolveDevice(requested: 'auto' | number | 'cpu' | undefined, requireGpu: boolean, allowCpu: boolean): DeviceRef {
    const gpu = this.gpuCache
    if (requested === 'cpu') return { kind: 'cpu' }
    if (typeof requested === 'number') {
      const found = gpu.gpus.find((entry) => entry.index === requested)
      if (found) return { kind: 'gpu', index: found.index, name: found.name }
      if (requireGpu || !allowCpu) {
        throw errWithCode(`GPU ${requested} is not available`, 'gpu-unavailable', {
          gpus: gpu.gpus.map((entry) => entry.index),
          error: gpu.error,
        })
      }
      return { kind: 'cpu' }
    }
    if (gpu.available && gpu.gpus.length) {
      const best = [...gpu.gpus].sort((a, b) => b.freeMB - a.freeMB)[0]
      if (best) return { kind: 'gpu', index: best.index, name: best.name }
    }
    if (requireGpu) {
      throw errWithCode('No GPU is available to the GPU tool service', 'gpu-unavailable', {
        error: gpu.error,
        hints: gpu.hints,
      })
    }
    if (!allowCpu) {
      throw errWithCode('GPU is required for this job', 'gpu-unavailable', { error: gpu.error, hints: gpu.hints })
    }
    return { kind: 'cpu' }
  }

  private deviceIndex(device: DeviceRef): number | null {
    return device.kind === 'gpu' ? (device.index ?? 0) : null
  }

  private jobInput(meta: SubmitMeta, operation: string, params: Record<string, unknown>, device: DeviceRef, argv: string[], engine: string | null, kind: JobRecord['kind']) {
    return {
      kind,
      operation,
      params,
      device,
      engine,
      argv,
      tool: meta.tool,
      argsHash: meta.argsHash,
      label: meta.label,
      clientName: meta.clientName,
      who: meta.who,
    }
  }

  async probe(path: string): Promise<unknown> {
    return await probeMedia(path, this.settingsStore.get())
  }

  /** Ставит задание ffmpeg; возможности проверяются один раз до очереди. */
  async submitFfmpeg(params: FfmpegParams, meta: SubmitMeta): Promise<JobRecord> {
    const settings = this.settingsStore.get()
    if (params.operation !== 'raw' && params.input && !existsSync(params.input)) {
      throw errWithCode(`Input not found: ${params.input}`, 'input-missing')
    }
    if (params.operation === 'concat') {
      for (const input of params.inputs ?? []) {
        if (!existsSync(input)) throw errWithCode(`Input not found: ${input}`, 'input-missing')
      }
    }
    const device = this.resolveDevice(params.device, false, settings.allowCpuFallback)
    const gpuBinary = device.kind === 'gpu'
    const caps = await this.caps(gpuBinary)
    const outputs = this.expectedOutputs(params)
    if (!params.overwrite) {
      for (const path of outputs) {
        if (existsSync(path)) throw errWithCode(`Output already exists: ${path}`, 'output-exists')
      }
    }
    let probePromise: Promise<unknown> | null = null
    if (params.operation !== 'raw' && params.operation !== 'concat' && params.input) {
      probePromise = probeMedia(params.input, settings).catch(() => null)
    }
    const argsHash = meta.argsHash || hashArgs(params)
    // План собирается внутри исполнителя: результаты probe и caps уже готовы.
    const jobPromise = probePromise ?? Promise.resolve(null)
    const manager = this.jobs
    const self = this
    const record = manager.submit(
      this.jobInput(meta, params.operation, params as unknown as Record<string, unknown>, device, [], gpuBinary ? settings.ffmpegGpuPath : settings.ffmpegPath, 'ffmpeg'),
      async (ctx) => {
        const probe = (await jobPromise) as ProbeInfo | null
        const plan = buildFfmpegPlan(params, caps, settings, probe, self.deviceIndex(device))
        ctx.job.argv = plan.argv
        ctx.job.engine = plan.binary
        for (const warning of plan.warnings) ctx.log(`warning: ${warning}`)
        ctx.log(`argv: ${plan.argv.join(' ')}`)
        if (plan.concatList) {
          await ensureParent(plan.concatList.path)
          const { writeFile } = await import('node:fs/promises')
          await writeFile(plan.concatList.path, plan.concatList.content, 'utf8')
        }
        if (params.operation === 'frames' && params.outputDir) await ensureOutputDir(params.outputDir)
        if (params.output) await ensureParent(params.output)
        if (params.operation === 'split' && params.outputDir) await ensureOutputDir(params.outputDir)
        ctx.phase('running', plan.operations.join(','))
        const state: ProgressState = { outTimeUs: null, frame: null, fps: null, speed: null, totalSize: null, done: false }
        let lastEmit = 0
        // ffmpeg получает host-индекс через -gpu/-hwaccel_device, без CUDA_VISIBLE_DEVICES.
        const env = { ...gpuEnv(settings, undefined) }
        delete env.CUDA_VISIBLE_DEVICES
        const result = await stream(plan.binary, plan.argv, {
          env,
          signal: ctx.signal,
          onStdoutLine: (line) => {
            if (!applyProgressLine(state, line)) return
            const now = Date.now()
            if (now - lastEmit < 120 && !state.done) return
            lastEmit = now
            const processedMs = state.outTimeUs != null ? Math.round(state.outTimeUs / 1000) : null
            const totalMs = plan.totalMs
            const percent = processedMs != null && totalMs ? Math.min(99.9, (processedMs / totalMs) * 100) : null
            const etaMs = state.speed && state.speed > 0 && processedMs != null && totalMs ? Math.max(0, Math.round((totalMs - processedMs) / state.speed)) : null
            ctx.progress({ processedMs, totalMs, percent, etaMs, fps: state.fps, speed: state.speed, frame: state.frame, bytes: state.totalSize })
          },
          onStderrLine: (line) => ctx.log(line),
          tailLines: 300,
        })
        if (result.code !== 0) {
          const tail = result.tail.slice(-8).join(' | ')
          throw errWithCode(`ffmpeg exited ${result.code}${tail ? `: ${tail}` : ''}`, 'ffmpeg-failed', { code: result.code, tail })
        }
        const collected = await self.collectOutputs(params, plan)
        for (const entry of collected) ctx.output(entry)
        ctx.log(`outputs: ${collected.map((entry) => entry.path).join(', ') || 'none'}`)
      },
    )
    return record
  }

  private expectedOutputs(params: FfmpegParams): string[] {
    if (params.operation === 'split') return []
    if (params.operation === 'frames') return []
    if (params.operation === 'raw') return params.args && params.args.includes('-y') ? [] : []
    return params.output ? [params.output] : []
  }

  private async collectOutputs(params: FfmpegParams, plan: { totalMs: number | null }): Promise<JobOutput[]> {
    const paths: string[] = []
    if (params.operation === 'frames' && params.outputDir) {
      const entries = await readdir(params.outputDir).catch(() => [] as string[])
      for (const entry of entries) if (/^frame_\d+\.(png|jpg)$/.test(entry)) paths.push(join(params.outputDir, entry))
    } else if (params.operation === 'split') {
      const pattern = params.output ?? (params.outputDir ? join(params.outputDir, 'segment_%03d.mp4') : '')
      const dir = dirname(pattern)
      const base = pattern.slice(dir.length + 1)
      const escaped = base.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/%0?\d*d/g, '\\d+')
      const matcher = new RegExp(`^${escaped}$`)
      const entries = await readdir(dir).catch(() => [] as string[])
      for (const entry of entries) if (matcher.test(entry)) paths.push(join(dir, entry))
    } else if (params.output) {
      paths.push(params.output)
    }
    const results: JobOutput[] = []
    for (const path of paths) {
      const info = await stat(path).catch(() => null)
      if (!info?.isFile()) continue
      results.push({ path, bytes: info.size, durationSec: plan.totalMs != null ? plan.totalMs / 1000 : null })
    }
    return results
  }

  submitYolo(params: Record<string, unknown>, meta: SubmitMeta): JobRecord {
    const settings = this.settingsStore.get()
    const model = typeof params.model === 'string' && params.model ? resolveModel(settings, params.model) : join(settings.modelsDir, 'yolo11n.pt')
    const source = typeof params.source === 'string' ? resolvePath(params.source) : ''
    if (!source) throw errWithCode('source is required', 'invalid-args')
    if (!existsSync(source)) throw errWithCode(`Source not found: ${source}`, 'input-missing')
    const device = this.resolveDevice(params.device as 'auto' | number | 'cpu' | undefined, true, false)
    const self = this
    return this.jobs.submit(
      this.jobInput(meta, 'yolo', params, device, [], 'ultralytics', 'yolo'),
      async (ctx) => {
        const outputDir = typeof params.outputDir === 'string' && params.outputDir ? resolvePath(params.outputDir) : join(settings.outputRoot, `yolo-${ctx.job.id}`)
        await ensureOutputDir(outputDir)
        const config = {
          model,
          source,
          track: params.track === true,
          tracker: typeof params.tracker === 'string' ? params.tracker : 'botsort.yaml',
          imgsz: numberOrUndefined(params.imgsz) ?? 640,
          conf: numberOrUndefined(params.conf) ?? 0.25,
          iou: numberOrUndefined(params.iou) ?? 0.45,
          classes: Array.isArray(params.classes) ? params.classes : undefined,
          half: params.half !== false,
          saveAnnotated: params.saveAnnotated === true,
          maxFrames: numberOrUndefined(params.maxFrames),
          device: self.deviceIndex(device) != null ? 'cuda:0' : 'cpu',
          outputDir,
        }
        ctx.job.params = { ...params, resolvedModel: model, outputDir }
        const value = await runPythonWorker({
          script: pythonScript('yolo_worker.py'),
          config,
          settings,
          ctx,
          deviceIndex: self.deviceIndex(device),
        })
        const result = value as { jsonl?: string; annotatedDir?: string; frames?: number } | null
        if (result?.jsonl) {
          const info = await stat(result.jsonl).catch(() => null)
          if (info) ctx.output({ path: result.jsonl, bytes: info.size, durationSec: null })
        }
        if (result?.annotatedDir) {
          const info = await stat(result.annotatedDir).catch(() => null)
          if (info?.isDirectory()) ctx.output({ path: result.annotatedDir, bytes: null, durationSec: null })
        }
      },
    )
  }

  submitOnnx(params: Record<string, unknown>, meta: SubmitMeta): JobRecord {
    const settings = this.settingsStore.get()
    const model = typeof params.model === 'string' ? resolveModel(settings, params.model) : ''
    if (!model || !existsSync(model)) throw errWithCode(`Model not found: ${model || '(none)'}`, 'onnx-model-missing')
    const inputs = Array.isArray(params.inputs) ? params.inputs : []
    if (!inputs.length) throw errWithCode('inputs is required', 'invalid-args')
    for (const entry of inputs as Array<Record<string, unknown>>) {
      const path = typeof entry?.path === 'string' ? entry.path : ''
      if (!path || !existsSync(path)) throw errWithCode(`Input array not found: ${path || '(none)'}`, 'input-missing')
    }
    const requireGpu = params.requireGpu !== false && params.device !== 'cpu'
    const device = this.resolveDevice(params.device as 'auto' | number | 'cpu' | undefined, requireGpu, !requireGpu)
    const self = this
    return this.jobs.submit(
      this.jobInput(meta, 'onnx', params, device, [], 'onnxruntime', 'onnx'),
      async (ctx) => {
        const outputDir = typeof params.outputDir === 'string' && params.outputDir ? resolvePath(params.outputDir) : join(settings.outputRoot, `onnx-${ctx.job.id}`)
        await ensureOutputDir(outputDir)
        const value = await runPythonWorker({
          script: pythonScript('onnx_worker.py'),
          config: {
            model,
            inputs,
            requireGpu,
            outputDir,
            providers: params.providers,
            device: self.deviceIndex(device) != null ? 'cuda:0' : 'cpu',
          },
          settings,
          ctx,
          deviceIndex: self.deviceIndex(device),
        })
        const info = await stat(outputDir).catch(() => null)
        if (info?.isDirectory()) ctx.output({ path: outputDir, bytes: null, durationSec: null })
        ctx.log(`onnx result: ${JSON.stringify(value)}`)
      },
    )
  }

  submitFace(params: Record<string, unknown>, meta: SubmitMeta): JobRecord {
    const settings = this.settingsStore.get()
    const source = typeof params.source === 'string' ? resolvePath(params.source) : ''
    if (!source || !existsSync(source)) throw errWithCode(`Source not found: ${source || '(none)'}`, 'input-missing')
    const device = this.resolveDevice(params.device as 'auto' | number | 'cpu' | undefined, params.device !== 'cpu', true)
    const fps = numberOrUndefined(params.fps) ?? 1
    const self = this
    return this.jobs.submit(
      this.jobInput(meta, 'face', params, device, [], 'onnxruntime/face', 'face'),
      async (ctx) => {
        const probe = await probeMedia(source, settings).catch(() => null)
        const outputDir = typeof params.outputDir === 'string' && params.outputDir ? resolvePath(params.outputDir) : join(settings.outputRoot, `face-${ctx.job.id}`)
        await ensureOutputDir(outputDir)
        const durationSec = numberOrUndefined(params.durationSec) ?? null
        const totalFrames = durationSec != null ? Math.max(1, Math.round(durationSec * fps)) : probe?.durationSec != null ? Math.max(1, Math.round(probe.durationSec * fps)) : null
        ctx.progress({ totalFrames, fps: null, percent: 0 })
        const backend = (typeof params.worker === 'string' ? params.worker : settings.faceWorker) || 'auto'
        const availability = rustFaceAvailable(settings)
        const useRust = backend === 'rust' || (backend === 'auto' && availability.ok)
        if (useRust && availability.ok) {
          try {
            if (!self.faceWorker) {
              self.faceWorker = new RustFaceWorker(settings, self.deviceIndex(device) ?? undefined, (line) => ctx.log(line))
            }
            const result = await self.faceWorker.run(
              {
                source,
                refImage: typeof params.refImage === 'string' ? resolvePath(params.refImage) : undefined,
                fps,
                startSec: numberOrUndefined(params.startSec) ?? 0,
                durationSec,
                width: probe?.video?.width ?? 1280,
                height: probe?.video?.height ?? 720,
                conf: numberOrUndefined(params.conf) ?? 0.6,
                matchThreshold: numberOrUndefined(params.matchThreshold) ?? 0.363,
                outputDir,
                modelDir: settings.modelsDir,
                ffmpegPath: settings.ffmpegPath,
                batch: numberOrUndefined(params.batch) ?? 8,
                threads: numberOrUndefined(params.threads) ?? 2,
              },
              ctx,
            )
            if (result.jsonl) {
              const info = await stat(result.jsonl).catch(() => null)
              if (info) ctx.output({ path: result.jsonl, bytes: info.size, durationSec: null })
            }
            ctx.log(`face result: ${JSON.stringify(result)}`)
            return
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error)
            if (backend === 'rust') throw error
            ctx.log(`rust face worker failed, falling back to python: ${message}`)
            self.faceWorker = null
          }
        } else if (backend === 'rust') {
          throw errWithCode(`Rust face worker is unavailable: ${availability.reason}`, 'face-worker-unavailable')
        }
        const value = await runPythonWorker({
          script: pythonScript('face_worker.py'),
          config: {
            source,
            ffmpegPath: settings.ffmpegPath,
            fps,
            startSec: numberOrUndefined(params.startSec) ?? 0,
            durationSec,
            width: probe?.video?.width ?? 1280,
            height: probe?.video?.height ?? 720,
            totalFrames,
            modelDir: settings.modelsDir,
            refImage: typeof params.refImage === 'string' ? resolvePath(params.refImage) : undefined,
            conf: numberOrUndefined(params.conf) ?? 0.6,
            matchThreshold: numberOrUndefined(params.matchThreshold) ?? 0.363,
            device: self.deviceIndex(device) != null ? 'cuda:0' : 'cpu',
            outputDir,
          },
          settings,
          ctx,
          deviceIndex: self.deviceIndex(device),
        })
        const result = value as { jsonl?: string } | null
        if (result?.jsonl) {
          const info = await stat(result.jsonl).catch(() => null)
          if (info) ctx.output({ path: result.jsonl, bytes: info.size, durationSec: null })
        }
        ctx.log(`face result: ${JSON.stringify(value)}`)
      },
    )
  }

  submitOcr(params: Record<string, unknown>, meta: SubmitMeta): JobRecord {
    const settings = this.settingsStore.get()
    const images = Array.isArray(params.images) ? (params.images as unknown[]).filter((item): item is string => typeof item === 'string') : []
    const source = typeof params.source === 'string' ? resolvePath(params.source) : ''
    if (!source && !images.length) throw errWithCode('source or images is required', 'invalid-args')
    if (source && !existsSync(source)) throw errWithCode(`Source not found: ${source}`, 'input-missing')
    for (const image of images) if (!existsSync(image)) throw errWithCode(`Image not found: ${image}`, 'input-missing')
    const requested = typeof params.engine === 'string' ? params.engine : settings.ocrEngine || 'auto'
    const engine = requested === 'deepseek' || requested === 'local' ? requested : 'auto'
    const self = this
    // DeepSeek OCR работает в облаке и не занимает GPU: задание идёт в CPU-полосу.
    const device: DeviceRef = engine === 'local' ? this.resolveDevice(params.device as 'auto' | number | 'cpu' | undefined, params.device !== 'cpu', true) : { kind: 'cpu' }
    return this.jobs.submit(
      this.jobInput(meta, 'ocr', { ...params, engine }, device, [], engine === 'local' ? 'onnxruntime/ocr' : 'deepseek/ocr', 'ocr'),
      async (ctx) => {
        const preferred: Array<'deepseek' | 'local'> = engine === 'auto' ? ['deepseek', 'local'] : [engine]
        for (const candidate of preferred) {
          if (candidate === 'deepseek') {
            try {
              await self.runDeepseekOcr(ctx, settings, params, source, images)
              return
            } catch (error) {
              const message = self.redact(error, '')
              // Ключ не настроен — это нормальный режим без DeepSeek, а не сбой.
              const skipped = error instanceof DeepseekOcrError && error.code === 'api-key-missing'
              ctx.log(`deepseek ocr ${skipped ? 'skipped' : 'failed'}: ${message}`)
              if (engine === 'deepseek') throw errWithCode(`DeepSeek OCR failed: ${message}`, 'deepseek-ocr-failed')
              const deviceRef = self.resolveDevice(params.device as 'auto' | number | 'cpu' | undefined, params.device !== 'cpu', true)
              ctx.job.device = deviceRef
              ctx.log(`fallback to local PP-OCRv5 on ${describeDevice(deviceRef)}`)
            }
          } else {
            await self.runLocalOcr(ctx, settings, params, source, images, self.resolveDevice(params.device as 'auto' | number | 'cpu' | undefined, params.device !== 'cpu', true))
            return
          }
        }
      },
    )
  }

  /** Ключ в тексте ошибки заменяется пометкой. */
  private redact(error: unknown, apiKey: string): string {
    const message = error instanceof Error ? error.message : String(error)
    const secret = (apiKey ?? '').trim()
    return (secret ? message.split(secret).join('[redacted]') : message).slice(0, 600)
  }

  /** OCR через DeepSeek Tools API и запись результата в JSONL. */
  async runDeepseekOcr(
    ctx: JobContext,
    settings: Settings,
    params: Record<string, unknown>,
    source: string,
    images: string[],
  ): Promise<DeepseekOcrResult> {
    ctx.phase('auth', 'резолв ключа DeepSeek')
    const apiKey = await this.deepseekKey(settings)
    if (!apiKey) throw new DeepseekOcrError('api-key-missing', 'DeepSeek API key is not configured')
    const purpose = params.purpose === 'rich_layout_recognition' ? 'rich_layout_recognition' : 'text_recognition'
    const targets = source ? [source] : images
    const outputDir = typeof params.outputDir === 'string' && params.outputDir ? resolvePath(params.outputDir) : join(settings.outputRoot, `ocr-${ctx.job.id}`)
    await ensureOutputDir(outputDir)
    const jsonl = join(outputDir, 'ocr.jsonl')
    const lines: string[] = []
    ctx.progress({ totalFrames: targets.length, frame: 0, percent: 0 })
    let last: DeepseekOcrResult | null = null
    for (const [order, target] of targets.entries()) {
      ctx.phase('upload', `DeepSeek OCR ${order + 1}/${targets.length}: ${basename(target)}`)
      last = await runDeepseekOcr({
        apiKey,
        baseUrl: settings.deepseekBaseUrl,
        file: target,
        purpose,
        timeoutMs: settings.deepseekTimeoutMs,
        signal: ctx.signal,
      })
      ctx.log(`deepseek ocr ${basename(target)}: ${last.pageCount} стр., ${last.usage.totalTokens} токенов, ${last.elapsedMs} мс`)
      for (const page of last.pages) {
        const record = {
          engine: 'deepseek',
          file: target,
          page: page.index,
          text: page.text,
          finishReason: page.finishReason,
          ocrId: last.id,
          purpose: last.purpose,
        }
        lines.push(JSON.stringify(record))
        if (page.finishReason === 'length') ctx.log(`warning: page ${page.index} of ${basename(target)} was truncated (finish_reason=length)`)
      }
      ctx.progress({ frame: order + 1, percent: Math.round(((order + 1) / targets.length) * 100) })
    }
    await Engine.ocrFileWriter(jsonl, lines.join('\n') + '\n', 'utf8')
    const info = await stat(jsonl).catch(() => null)
    if (info) ctx.output({ path: jsonl, bytes: info.size, durationSec: null })
    ctx.log(
      `ocr result: ${JSON.stringify({
        engine: 'deepseek',
        files: targets.length,
        pages: last?.pageCount ?? 0,
        tokens: last?.usage.totalTokens ?? 0,
        elapsedMs: last?.elapsedMs ?? 0,
        jsonl,
      })}`,
    )
    return last as DeepseekOcrResult
  }

  /** Локальный PP-OCRv5 на CUDA: прежний путь, оставлен как запасной. */
  async runLocalOcr(
    ctx: JobContext,
    settings: Settings,
    params: Record<string, unknown>,
    source: string,
    images: string[],
    device: DeviceRef,
  ): Promise<void> {
    const outputDir = typeof params.outputDir === 'string' && params.outputDir ? resolvePath(params.outputDir) : join(settings.outputRoot, `ocr-${ctx.job.id}`)
    await ensureOutputDir(outputDir)
    let durationSec = numberOrUndefined(params.durationSec) ?? null
    let width = 1280
    let height = 720
    if (source) {
      const probe = await probeMedia(source, settings).catch(() => null)
      if (probe?.video?.width) width = probe.video.width
      if (probe?.video?.height) height = probe.video.height
      if (durationSec == null) durationSec = probe?.durationSec ?? null
    }
    const fps = numberOrUndefined(params.fps) ?? 1
    if (durationSec != null && !params.atSec) {
      ctx.progress({ totalFrames: Math.max(1, Math.round(durationSec * fps)), percent: 0 })
    }
    ctx.phase('local', 'локальный PP-OCRv5 на GPU')
    const value = await runPythonWorker({
      script: pythonScript('ocr_worker.py'),
      config: {
        modelDir: settings.ocrModelDir,
        dictPath: typeof params.dictPath === 'string' ? params.dictPath : undefined,
        region: params.region ?? (settings.ocrRegion || undefined),
        source: source || undefined,
        images,
        atSec: Array.isArray(params.atSec) ? params.atSec : undefined,
        fps,
        startSec: numberOrUndefined(params.startSec) ?? 0,
        durationSec,
        width,
        height,
        detLimit: numberOrUndefined(params.detLimit) ?? 960,
        outputDir,
        ffmpegPath: settings.ffmpegPath,
        device: this.deviceIndex(device) != null ? 'cuda:0' : 'cpu',
      },
      settings,
      ctx,
      deviceIndex: this.deviceIndex(device),
    })
    const result = value as { jsonl?: string } | null
    if (result?.jsonl) {
      const info = await stat(result.jsonl).catch(() => null)
      if (info) ctx.output({ path: result.jsonl, bytes: info.size, durationSec: null })
    }
    ctx.log(`ocr result: ${JSON.stringify(value)}`)
  }

  submitPython(params: Record<string, unknown>, meta: SubmitMeta): JobRecord {
    const settings = this.settingsStore.get()
    const script = typeof params.script === 'string' ? resolvePath(params.script) : ''
    if (!script || !existsSync(script)) throw errWithCode(`Script not found: ${script || '(none)'}`, 'input-missing')
    const device = this.resolveDevice(params.device as 'auto' | number | 'cpu' | undefined, false, true)
    const self = this
    return this.jobs.submit(
      this.jobInput(meta, 'python', params, device, [], 'python', 'python'),
      async (ctx) => {
        const args = Array.isArray(params.args) ? (params.args as unknown[]).map((item) => String(item)) : []
        const cwd = typeof params.cwd === 'string' ? params.cwd : undefined
        const extraEnv = params.env && typeof params.env === 'object' ? (params.env as Record<string, string>) : {}
        ctx.log(`python ${script} ${args.join(' ')}`)
        const state = { percent: null as number | null }
        const result = await stream(settings.pythonPath, [script, ...args], {
          cwd,
          env: { ...gpuEnv(settings, self.deviceIndex(device) ?? undefined), ...extraEnv },
          signal: ctx.signal,
          onStdoutLine: (line) => {
            ctx.log(line)
            const match = /progress[^0-9]{0,12}(\d{1,3})\s*%/i.exec(line)
            if (match) {
              const percent = Math.max(0, Math.min(100, Number(match[1])))
              state.percent = percent
              ctx.progress({ percent, phase: 'running' })
            }
          },
          onStderrLine: (line) => ctx.log(line),
          tailLines: 300,
        })
        if (result.code !== 0) {
          throw errWithCode(`python exited ${result.code}`, 'python-failed', { tail: result.tail.slice(-8) })
        }
      },
    )
  }

  submitSetup(action: string, meta: SubmitMeta): JobRecord {
    const settings = this.settingsStore.get()
    const device: DeviceRef = { kind: 'cpu' }
    const self = this
    return this.jobs.submit(
      this.jobInput(meta, `setup:${action}`, { action }, device, [], 'setup', 'setup'),
      async (ctx) => {
        const venv = files.venv()
        const python = join(venv, 'bin', 'python')
        const uv = settings.uvPath || 'uv'
        const step = async (phase: string, command: string, args: string[], timeoutMs = 1_800_000): Promise<void> => {
          ctx.phase(phase)
          ctx.log(`$ ${command} ${args.join(' ')}`)
          const result = await stream(command, args, {
            signal: ctx.signal,
            env: { ...process.env, UV_CACHE_DIR: join(files.home(), '.uv-cache'), UV_LINK_MODE: 'copy' },
            onStdoutLine: (line) => ctx.log(line),
            onStderrLine: (line) => ctx.log(line),
            tailLines: 200,
            timeoutMs,
          })
          if (result.code !== 0) {
            throw errWithCode(`${phase} failed (${command} exited ${result.code})`, 'setup-failed', { phase, tail: result.tail.slice(-10) })
          }
        }
        if (action === 'install-runtime') {
          if (!existsSync(python)) await step('venv', uv, ['venv', '--python', '3.12', venv])
          await step('torch', uv, ['pip', 'install', '--python', python, '--upgrade', 'torch', 'torchvision', '--index-url', settings.torchIndexUrl])
          // lap нужен трекеру BoT-SORT (YOLO track), остальные пакеты — воркерам.
          await step('packages', uv, ['pip', 'install', '--python', python, '--upgrade', 'onnxruntime-gpu', 'ultralytics', 'opencv-python-headless', 'numpy', 'lap>=0.5.12'])
        } else if (action === 'install-torch-cuda') {
          if (!existsSync(python)) throw errWithCode('GPU venv is missing; run install-runtime first', 'setup-failed')
          await step('torch', uv, ['pip', 'install', '--python', python, '--upgrade', 'torch', 'torchvision', '--index-url', settings.torchIndexUrl])
        } else if (action === 'install-onnx-gpu') {
          if (!existsSync(python)) throw errWithCode('GPU venv is missing; run install-runtime first', 'setup-failed')
          await step('onnxruntime', uv, ['pip', 'install', '--python', python, '--upgrade', 'onnxruntime-gpu'])
        } else if (action === 'refresh-ffmpeg') {
          self.doctor.invalidate()
          ctx.log('ffmpeg capability cache cleared; doctor will re-probe on next request')
        } else {
          throw errWithCode(`Unknown setup action: ${action}`, 'invalid-args')
        }
        if (existsSync(python) && action !== 'refresh-ffmpeg') {
          const probe = await pythonProbe({ ...settings, pythonPath: python })
          ctx.log(`verify: ${JSON.stringify(probe)}`)
          if (!probe) throw errWithCode('Installed runtime cannot be imported', 'setup-verify-failed')
          const next = await self.settingsStore.update({ pythonPath: python })
          self.doctor.invalidate()
          ctx.log(`settings.pythonPath -> ${next.pythonPath}`)
          if (action === 'install-runtime' && probe && !probe.torchCuda) {
            ctx.log('warning: torch.cuda.is_available() is false; CUDA will not be used until the GPU is visible to this process')
          }
        }
      },
    )
  }

  listJobs(limit = 50, activeOnly = false): JobRecord[] {
    return this.jobs.list(limit, activeOnly)
  }

  getJob(id: string): JobRecord | null {
    return this.jobs.get(id)
  }

  cancelJob(id: string): boolean {
    return this.jobs.cancel(id)
  }

  addCall(record: Omit<CallRecord, 'id'>): CallRecord {
    return this.jobs.addCall(record)
  }

  addHint(input: { tool: string; argsHash: string; who: Who }): { id: string } {
    const hint = this.jobs.addHint(input)
    return { id: hint.id }
  }

  peekWho(tool: string, argsHash: string): Who | null {
    return this.jobs.peekHint(tool, argsHash)?.who ?? null
  }

  calls(limit = 100): CallRecord[] {
    return this.jobs.callsList(limit)
  }

  history(query: HistoryQuery): { jobs: JobRecord[]; calls: CallRecord[] } {
    const limit = Math.max(1, Math.min(500, query.limit ?? 100))
    const jobs = this.jobs
      .list(500)
      .filter((job) => (query.kind ? job.kind === query.kind || job.operation === query.kind : true))
      .filter((job) => (query.sessionId ? job.who.sessionId === query.sessionId : true))
      .filter((job) => (query.operation ? job.operation === query.operation : true))
      .slice(0, limit)
    const calls = this.jobs
      .callsList(500)
      .filter((call) => (query.kind ? call.tool === query.kind || call.operation === query.kind : true))
      .filter((call) => (query.sessionId ? call.who.sessionId === query.sessionId : true))
      .slice(0, limit)
    return { jobs, calls }
  }

  leases(): LeaseRecord[] {
    return this.jobs.leasesList()
  }

  acquireLease(input: { holder: string; ttlSec: number; devices: number[]; who: Who }): LeaseRecord {
    return this.jobs.addLease(input)
  }

  releaseLease(id: string): boolean {
    return this.jobs.releaseLease(id)
  }

  describeDevice(device: DeviceRef): string {
    return describeDevice(device)
  }

  async logPath(jobId: string): Promise<string | null> {
    const job = this.jobs.get(jobId)
    return job?.logPath ?? null
  }

  hash(value: unknown): string {
    return hashArgs(value)
  }
}
