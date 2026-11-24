/**
 * GPU-only x2-апскейл (SPAN fp16): постоянный Rust-воркер `bin/gputool-upscale`.
 *
 * Инференс строго на CUDA: воркер создаёт сессию ORT с
 * `session.disable_cpu_ep_fallback`, модель обязана быть fp16 и давать ровно x2.
 * Кадры идут батчами, без тайлинга; видео — NVDEC → CUDA → NVENC.
 */
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { createInterface } from 'node:readline'
import type { JobContext, Settings } from './types.js'
import { gpuEnv } from './settings.js'
import { packageRoot } from './python.js'
import { onnxRuntimeLibrary } from './face-rust.js'

const IMAGE_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'webp', 'bmp'])

/** Путь к собранному бинарю апскейлера. */
export function upscaleBinary(): string {
  const override = process.env.GPUTOOL_UPSCALE_BIN
  if (override) return override
  return join(packageRoot(), 'bin', 'gputool-upscale')
}

/** Модель апскейла: параметр задания, настройка или modelsDir/upscale. */
export function upscaleModelPath(settings: Settings, requested?: string): string {
  if (requested) return requested
  if (settings.upscaleModelPath) return settings.upscaleModelPath
  return join(settings.modelsDir, 'upscale', 'span2x_ch48_fp16.onnx')
}

/** Готов ли GPU-путь: бинарник, ORT-библиотека и файл модели. */
export function upscaleAvailable(
  settings: Settings,
  model?: string,
): { ok: boolean; binary: string; library: string | null; model: string; reason: string } {
  const binary = upscaleBinary()
  const library = onnxRuntimeLibrary(settings)
  const modelPath = upscaleModelPath(settings, model)
  if (!existsSync(binary)) return { ok: false, binary, library, model: modelPath, reason: `binary not found: ${binary}` }
  if (!library) return { ok: false, binary, library, model: modelPath, reason: 'libonnxruntime.so is missing in the plugin venv' }
  if (!existsSync(modelPath)) return { ok: false, binary, library, model: modelPath, reason: `model not found: ${modelPath}` }
  return { ok: true, binary, library, model: modelPath, reason: 'ok' }
}

export interface UpscaleParams {
  source?: string
  sources?: string[]
  output?: string
  model?: string
  batch?: number
  provider?: 'auto' | 'cuda' | 'tensorrt'
  width?: number
  height?: number
  codec?: 'hevc' | 'h264'
  quality?: 'fast' | 'balanced' | 'quality'
  startSec?: number
  durationSec?: number
  fps?: number
  overwrite?: boolean
  device?: 'auto' | number | 'cpu'
}

export interface UpscaleConfig {
  source: string
  output: string
  model: string
  batch: number
  provider: 'auto' | 'cuda' | 'tensorrt'
  engineCache?: string
  width?: number
  height?: number
  overlay?: never
  codec?: 'hevc' | 'h264'
  quality?: string
  startSec?: number
  durationSec?: number
  fps?: number
  ffmpegPath: string
  ffprobePath: string
  overwrite: boolean
}

export interface UpscaleResult {
  engine: 'rust'
  device: string
  providers: string[]
  scale: number
  output: string
  width: number
  height: number
  bytes: number
  elapsedMs: number
  frames?: number
  fps?: number
  inferMsPerFrame?: number
}

/** Проверяет параметры и собирает конфигурацию для конкретного источника. */
export function buildUpscaleConfig(
  params: UpscaleParams,
  settings: Settings,
  source: string,
  output: string,
): UpscaleConfig {
  const batch = Math.round(params.batch ?? settings.upscaleBatch ?? 8)
  if (batch < 1 || batch > 64) {
    throw Object.assign(new Error('batch must be between 1 and 64'), { code: 'invalid-args' })
  }
  if ((params.width == null) !== (params.height == null)) {
    throw Object.assign(new Error('width and height must be set together'), { code: 'invalid-args' })
  }
  if (params.width != null && params.height != null) {
    if (!Number.isInteger(params.width) || !Number.isInteger(params.height) || params.width < 2 || params.height < 2) {
      throw Object.assign(new Error('width/height must be integers >= 2'), { code: 'invalid-args' })
    }
    if (params.width % 2 !== 0 || params.height % 2 !== 0) {
      throw Object.assign(new Error('width/height must be even for x2 upscaling'), { code: 'invalid-args' })
    }
  }
  if (!existsSync(source)) throw Object.assign(new Error(`Source not found: ${source}`), { code: 'input-missing' })
  const model = upscaleModelPath(settings, params.model)
  if (!existsSync(model)) throw Object.assign(new Error(`Model not found: ${model}`), { code: 'upscale-model-missing' })
  return {
    source,
    output,
    model,
    batch,
    provider: params.provider ?? settings.upscaleProvider ?? 'auto',
    engineCache: join(settings.modelsDir, 'upscale', 'trt-cache'),
    width: params.width,
    height: params.height,
    codec: params.codec ?? (settings.upscaleCodec as 'hevc' | 'h264') ?? 'hevc',
    quality: params.quality ?? settings.upscaleQuality ?? 'balanced',
    startSec: params.startSec,
    durationSec: params.durationSec,
    fps: params.fps,
    ffmpegPath: settings.ffmpegGpuPath,
    ffprobePath: settings.ffprobeGpuPath,
    overwrite: params.overwrite === true,
  }
}

/** Картинка или видео — от этого зависит режим воркера. */
export function isImageSource(path: string): boolean {
  const extension = path.slice(path.lastIndexOf('.') + 1).toLowerCase()
  return IMAGE_EXTENSIONS.has(extension)
}

interface Pending {
  resolve: (value: UpscaleResult) => void
  reject: (error: Error) => void
  ctx: JobContext
  emitted: number
}

/** Клиент постоянного процесса: сессия CUDA и модель переживают задания. */
export class RustUpscaleWorker {
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
    const availability = upscaleAvailable(settings)
    if (!availability.ok || !availability.library) throw new Error(`upscale worker unavailable: ${availability.reason}`)
    this.binary = availability.binary
    this.library = availability.library
  }

  async start(): Promise<void> {
    if (this.disposed) throw new Error('upscale worker is disposed')
    if (this.ready) return await this.ready
    this.ready = new Promise<void>((resolve, reject) => {
      const env = {
        ...gpuEnv(this.settings, this.deviceIndex),
        ORT_DYLIB_PATH: this.library,
      }
      const child = spawn(this.binary, ['--serve'], { env, stdio: ['pipe', 'pipe', 'pipe'] }) as ChildProcessWithoutNullStreams
      this.child = child
      let settled = false
      const timer = setTimeout(() => {
        if (settled) return
        settled = true
        reject(new Error('upscale worker did not start in time'))
      }, 30_000)
      timer.unref?.()
      createInterface({ input: child.stdout }).on('line', (line) => {
        let event: Record<string, unknown>
        try {
          event = JSON.parse(line) as Record<string, unknown>
        } catch {
          this.log(`upscale: ${line}`)
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
        if (text) this.log(`upscale stderr: ${text.slice(-800)}`)
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
        const reason = new Error(`upscale worker exited (code=${code ?? 'null'}, signal=${signal ?? 'null'})`)
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

  async run(config: UpscaleConfig, ctx: JobContext): Promise<UpscaleResult> {
    await this.start()
    if (!this.child) throw new Error('upscale worker is not running')
    if (this.child.pid != null) ctx.trackPid(this.child.pid)
    if (this.pending) throw new Error('upscale worker is busy')
    const id = ctx.job.id
    return await new Promise<UpscaleResult>((resolve, reject) => {
      this.pending = { resolve, reject, ctx, emitted: 0 }
      this.child?.stdin.write(`${JSON.stringify({ type: 'run', id, config })}\n`)
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
      pending.resolve({
        engine: 'rust',
        device: String(result.device ?? 'cuda:0'),
        providers: Array.isArray(result.providers) ? result.providers.map(String) : ['CUDAExecutionProvider'],
        scale: Number(result.scale ?? 2),
        output: String(result.output ?? ''),
        width: Number(result.width ?? 0),
        height: Number(result.height ?? 0),
        bytes: Number(result.bytes ?? 0),
        elapsedMs: Number(result.elapsedMs ?? 0),
        frames: result.frames != null ? Number(result.frames) : undefined,
        fps: result.fps != null ? Number(result.fps) : undefined,
        inferMsPerFrame: result.inferMsPerFrame != null ? Number(result.inferMsPerFrame) : undefined,
      })
      this.pending = null
      return
    }
    if (event.type === 'error') {
      const code = typeof event.code === 'string' ? event.code : 'upscale-failed'
      const message = typeof event.message === 'string' ? event.message : 'upscale worker failed'
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
