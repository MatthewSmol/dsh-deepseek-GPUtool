/**
 * Трекинг мяча настольного тенниса: постоянный Rust-воркер `bin/gputool-ball`.
 *
 * Модель — ONNX семейства TrackNetV2/V3 или WASB (3 кадра → heatmap). Инференс
 * идёт на CUDA, если EP доступен; при `device:"auto"` допустим CPU-фолбэк с
 * предупреждением (TrackNetV2 лёгкий). Выход: ball.jsonl/csv и видео с точкой.
 */
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { createInterface } from 'node:readline'
import type { JobContext, Settings } from './types.js'
import { gpuEnv } from './settings.js'
import { packageRoot } from './python.js'
import { onnxRuntimeLibrary } from './face-rust.js'

/** Путь к собранному бинарю трекинга мяча. */
export function ballBinary(): string {
  const override = process.env.GPUTOOL_BALL_BIN
  if (override) return override
  return join(packageRoot(), 'bin', 'gputool-ball')
}

/** Модель мяча: параметр задания, настройка или modelsDir/ball. */
export function ballModelPath(settings: Settings, requested?: string): string {
  if (requested) return requested
  if (settings.ballModelPath) return settings.ballModelPath
  return join(settings.modelsDir, 'ball', 'tracknetv2.onnx')
}

/** Готов ли путь: бинарник, ORT-библиотека и файл модели. */
export function ballAvailable(
  settings: Settings,
  model?: string,
): { ok: boolean; binary: string; library: string | null; model: string; reason: string } {
  const binary = ballBinary()
  const library = onnxRuntimeLibrary(settings)
  const modelPath = ballModelPath(settings, model)
  if (!existsSync(binary)) return { ok: false, binary, library, model: modelPath, reason: `binary not found: ${binary}` }
  if (!library) return { ok: false, binary, library, model: modelPath, reason: 'libonnxruntime.so is missing in the plugin venv' }
  if (!existsSync(modelPath)) return { ok: false, binary, library, model: modelPath, reason: `model not found: ${modelPath}` }
  return { ok: true, binary, library, model: modelPath, reason: 'ok' }
}

export interface BallParams {
  source?: string
  model?: string
  outputDir?: string
  fps?: number
  startSec?: number
  durationSec?: number
  threshold?: number
  annotate?: boolean
  maxGapFrames?: number
  smooth?: boolean
  maxJumpPx?: number
  predictOffset?: number
  outputIndex?: number
  sigmoid?: boolean
  frames?: number
  inputWidth?: number
  inputHeight?: number
  device?: 'auto' | number | 'cpu'
  overwrite?: boolean
}

export interface BallConfig {
  source: string
  outputDir: string
  model: string
  fps?: number
  startSec?: number
  durationSec?: number
  threshold: number
  annotate: boolean
  maxGapFrames: number
  smooth: boolean
  maxJumpPx?: number
  predictOffset: number
  outputIndex?: number
  sigmoid?: boolean
  frames?: number
  inputWidth?: number
  inputHeight?: number
  device: 'auto' | 'cpu' | 'cuda'
  gpuIndex?: number
  ffmpegPath: string
  ffprobePath: string
  overwrite: boolean
}

export interface BallResult {
  engine: 'rust'
  device: string
  providers: string[]
  frames: number
  detected: number
  fps: number
  hasAudio: boolean
  jsonl: string
  csv: string
  ass: string
  video: string | null
  model: { frames: number; width: number; height: number; offset: number }
  elapsedMs: number
}

/** Проверяет параметры и собирает конфигурацию воркера. */
export function buildBallConfig(
  params: BallParams,
  settings: Settings,
  source: string,
  outputDir: string,
  device: 'auto' | 'cpu' | 'cuda',
  gpuIndex?: number,
): BallConfig {
  if (!existsSync(source)) throw Object.assign(new Error(`Source not found: ${source}`), { code: 'input-missing' })
  const model = ballModelPath(settings, params.model)
  if (!existsSync(model)) throw Object.assign(new Error(`Model not found: ${model}`), { code: 'ball-model-missing' })
  if (params.threshold != null && (params.threshold < 0 || params.threshold > 1)) {
    throw Object.assign(new Error('threshold must be between 0 and 1'), { code: 'invalid-args' })
  }
  if (params.fps != null && (params.fps <= 0 || params.fps > 60)) {
    throw Object.assign(new Error('fps must be between 1 and 60'), { code: 'invalid-args' })
  }
  if (params.maxGapFrames != null && (params.maxGapFrames < 0 || params.maxGapFrames > 120)) {
    throw Object.assign(new Error('maxGapFrames must be between 0 and 120'), { code: 'invalid-args' })
  }
  return {
    source,
    outputDir,
    model,
    fps: params.fps,
    startSec: params.startSec,
    durationSec: params.durationSec,
    threshold: params.threshold ?? settings.ballThreshold ?? 0.5,
    annotate: params.annotate ?? settings.ballAnnotate ?? true,
    maxGapFrames: params.maxGapFrames ?? settings.ballMaxGapFrames ?? 12,
    smooth: params.smooth ?? true,
    maxJumpPx: params.maxJumpPx,
    predictOffset: params.predictOffset ?? 0,
    outputIndex: params.outputIndex,
    sigmoid: params.sigmoid,
    frames: params.frames,
    inputWidth: params.inputWidth,
    inputHeight: params.inputHeight,
    device,
    gpuIndex,
    ffmpegPath: settings.ffmpegGpuPath,
    ffprobePath: settings.ffprobeGpuPath,
    overwrite: params.overwrite === true,
  }
}

interface Pending {
  resolve: (value: BallResult) => void
  reject: (error: Error) => void
  ctx: JobContext
}

/** Клиент постоянного процесса: сессия ORT и модель переиспользуются. */
export class RustBallWorker {
  private child: ChildProcessWithoutNullStreams | null = null
  private pending: Pending | null = null
  private ready: Promise<void> | null = null
  private disposed = false
  private readonly binary: string
  private readonly library: string

  constructor(
    private readonly settings: Settings,
    private readonly deviceIndex: number | undefined,
    private readonly log: (message: string) => void,
  ) {
    const availability = ballAvailable(settings)
    if (!availability.ok || !availability.library) throw new Error(`ball worker unavailable: ${availability.reason}`)
    this.binary = availability.binary
    this.library = availability.library
  }

  async start(): Promise<void> {
    if (this.disposed) throw new Error('ball worker is disposed')
    if (this.ready) return await this.ready
    this.ready = new Promise<void>((resolve, reject) => {
      const env = { ...gpuEnv(this.settings, this.deviceIndex), ORT_DYLIB_PATH: this.library }
      const child = spawn(this.binary, ['--serve'], { env, stdio: ['pipe', 'pipe', 'pipe'] }) as ChildProcessWithoutNullStreams
      this.child = child
      let settled = false
      const timer = setTimeout(() => {
        if (settled) return
        settled = true
        reject(new Error('ball worker did not start in time'))
      }, 30_000)
      timer.unref?.()
      createInterface({ input: child.stdout }).on('line', (line) => {
        let event: Record<string, unknown>
        try {
          event = JSON.parse(line) as Record<string, unknown>
        } catch {
          this.log(`ball: ${line}`)
          return
        }
        this.onEvent(event)
        if (event.type === 'ready' && !settled) {
          settled = true
          clearTimeout(timer)
          resolve()
        }
      })
      child.stderr.on('data', (chunk: Buffer) => {
        const text = chunk.toString('utf8').trim()
        if (text) this.log(`ball stderr: ${text.slice(-800)}`)
      })
      child.on('error', (error) => {
        if (!settled) {
          settled = true
          clearTimeout(timer)
          reject(error)
        }
        this.pending?.reject(error)
        this.pending = null
      })
      child.on('exit', (code, signal) => {
        this.child = null
        this.ready = null
        const reason = new Error(`ball worker exited (code=${code ?? 'null'}, signal=${signal ?? 'null'})`)
        this.pending?.reject(reason)
        this.pending = null
        if (!settled) {
          settled = true
          clearTimeout(timer)
          reject(reason)
        }
      })
    })
    return await this.ready
  }

  async run(config: BallConfig, ctx: JobContext): Promise<BallResult> {
    await this.start()
    if (!this.child) throw new Error('ball worker is not running')
    if (this.child.pid != null) ctx.trackPid(this.child.pid)
    if (this.pending) throw new Error('ball worker is busy')
    return await new Promise<BallResult>((resolve, reject) => {
      this.pending = { resolve, reject, ctx }
      this.child?.stdin.write(`${JSON.stringify({ type: 'run', id: ctx.job.id, config })}\n`)
    })
  }

  private onEvent(event: Record<string, unknown>): void {
    const pending = this.pending
    if (event.type === 'log') {
      const message = typeof event.message === 'string' ? event.message : ''
      if (message) (pending?.ctx.log ?? this.log)(message)
      return
    }
    if (!pending) return
    if (event.type === 'progress') {
      pending.ctx.progress({
        percent: typeof event.percent === 'number' ? event.percent : undefined,
        phase: typeof event.phase === 'string' ? event.phase : undefined,
        frame: typeof event.frame === 'number' ? event.frame : undefined,
        totalFrames: typeof event.totalFrames === 'number' ? event.totalFrames : undefined,
        fps: typeof event.fps === 'number' ? event.fps : undefined,
        etaMs: typeof event.etaMs === 'number' ? event.etaMs : undefined,
      })
      return
    }
    if (event.type === 'result') {
      const result = (event.result ?? {}) as Record<string, unknown>
      const model = (result.model ?? {}) as Record<string, unknown>
      pending.resolve({
        engine: 'rust',
        device: String(result.device ?? 'cpu'),
        providers: Array.isArray(result.providers) ? result.providers.map(String) : [],
        frames: Number(result.frames ?? 0),
        detected: Number(result.detected ?? 0),
        fps: Number(result.fps ?? 0),
        hasAudio: result.hasAudio === true,
        jsonl: String(result.jsonl ?? ''),
        csv: String(result.csv ?? ''),
        ass: String(result.ass ?? ''),
        video: typeof result.video === 'string' ? result.video : null,
        model: {
          frames: Number(model.frames ?? 3),
          width: Number(model.width ?? 0),
          height: Number(model.height ?? 0),
          offset: Number(model.offset ?? 0),
        },
        elapsedMs: Number(result.elapsedMs ?? 0),
      })
      this.pending = null
      return
    }
    if (event.type === 'error') {
      const code = typeof event.code === 'string' ? event.code : 'ball-failed'
      const message = typeof event.message === 'string' ? event.message : 'ball worker failed'
      pending.reject(Object.assign(new Error(message), { code }))
      this.pending = null
    }
  }

  async dispose(): Promise<void> {
    this.disposed = true
    const child = this.child
    this.child = null
    this.ready = null
    if (!child) return
    try {
      child.stdin.write('{"type":"shutdown"}\n')
    } catch {
      /* процесс мог завершиться */
    }
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        try {
          child.kill('SIGKILL')
        } catch {
          /* игнорируем */
        }
        resolve()
      }, 2_000)
      timer.unref?.()
      child.once('exit', () => {
        clearTimeout(timer)
        resolve()
      })
    })
  }
}
