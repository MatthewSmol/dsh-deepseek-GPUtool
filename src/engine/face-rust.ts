/**
 * Постоянный Rust-воркер лиц: один процесс на движок, сессии ONNX Runtime
 * создаются один раз, кадры готовятся и считаются батчами.
 *
 * Зачем: python-воркер тратит на кадр ~6,5 мс, но каждое задание заново
 * инициализирует ORT, cuDNN и модели (~0,9–1,5 с). Постоянный процесс убирает
 * этот старт, а Rust снимает GIL с подготовки кадров.
 */
import { existsSync, readdirSync } from 'node:fs'
import { basename, join } from 'node:path'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { createInterface } from 'node:readline'
import type { JobContext, Settings } from './types.js'
import { gpuEnv } from './settings.js'
import { packageRoot } from './python.js'

/** Путь к собранному бинарю: bin/gputool-face рядом с пакетом. */
export function rustFaceBinary(): string {
  const override = process.env.GPUTOOL_FACE_BIN
  if (override) return override
  return join(packageRoot(), 'bin', 'gputool-face')
}

/**
 * Библиотека ONNX Runtime для `ort` (load-dynamic). Берём ту, что уже стоит в
 * venv плагина: отдельная загрузка бинарей не нужна.
 */
export function onnxRuntimeLibrary(settings: Settings): string | null {
  const python = settings.pythonPath
  if (!python.includes('/bin/python')) return null
  const venv = python.slice(0, python.indexOf('/bin/python'))
  const lib = join(venv, 'lib')
  try {
    for (const entry of readdirSync(lib)) {
      if (!entry.startsWith('python')) continue
      const capi = join(lib, entry, 'site-packages', 'onnxruntime', 'capi')
      if (!existsSync(capi)) continue
      const file = readdirSync(capi).find((name) => name.startsWith('libonnxruntime.so'))
      if (file) return join(capi, file)
    }
  } catch {
    /* venv без onnxruntime — Rust-путь недоступен */
  }
  return null
}

/** Доступен ли Rust-путь на этой машине. */
export function rustFaceAvailable(settings: Settings): { ok: boolean; binary: string; library: string | null; reason: string } {
  const binary = rustFaceBinary()
  const library = onnxRuntimeLibrary(settings)
  if (!existsSync(binary)) return { ok: false, binary, library, reason: `binary not found: ${binary}` }
  if (!library) return { ok: false, binary, library, reason: 'libonnxruntime.so is missing in the plugin venv' }
  return { ok: true, binary, library, reason: 'ok' }
}

export interface FaceRustConfig {
  source: string
  refImage?: string
  fps: number
  startSec: number
  durationSec: number | null
  width: number
  height: number
  conf: number
  matchThreshold: number
  outputDir: string
  modelDir: string
  ffmpegPath: string
  batch: number
  threads: number
}

export interface FaceRustResult {
  frames: number
  faces: number
  matches: number
  jsonl: string
  engine: 'rust'
  device: string
}

interface Pending {
  resolve: (value: FaceRustResult) => void
  reject: (error: Error) => void
  ctx: JobContext
  startedAt: number
  emitted: number
}

/**
 * Клиент постоянного процесса. Одно задание за раз: GPU-полоса движка и так
 * сериализует задания, а очередь внутри процесса не нужна.
 */
export class RustFaceWorker {
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
    const availability = rustFaceAvailable(settings)
    if (!availability.ok || !availability.library) throw new Error(`rust face worker unavailable: ${availability.reason}`)
    this.binary = availability.binary
    this.library = availability.library
  }

  /** Запускает процесс (идемпотентно) и ждёт готовности моделей. */
  async start(): Promise<void> {
    if (this.disposed) throw new Error('rust face worker is disposed')
    if (this.ready) return await this.ready
    this.ready = new Promise<void>((resolve, reject) => {
      const env = {
        ...gpuEnv(this.settings, this.deviceIndex),
        // ort с load-dynamic ищет библиотеку по этому пути.
        ORT_DYLIB_PATH: this.library,
      }
      const child = spawn(this.binary, ['--serve'], { env, stdio: ['pipe', 'pipe', 'pipe'] }) as ChildProcessWithoutNullStreams
      this.child = child
      let settled = false
      const timer = setTimeout(() => {
        if (settled) return
        settled = true
        reject(new Error('rust face worker did not start in time'))
      }, 20_000)
      timer.unref?.()
      const lines = createInterface({ input: child.stdout })
      lines.on('line', (line) => {
        let event: Record<string, unknown>
        try {
          event = JSON.parse(line) as Record<string, unknown>
        } catch {
          this.log(`rust face: ${line}`)
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
        if (text) this.log(`rust face stderr: ${text.slice(-800)}`)
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
        const reason = new Error(`rust face worker exited (code=${code ?? 'null'}, signal=${signal ?? 'null'})`)
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

  /** Выполняет одно задание и возвращает итог воркера. */
  async run(config: FaceRustConfig, ctx: JobContext): Promise<FaceRustResult> {
    await this.start()
    if (!this.child) throw new Error('rust face worker is not running')
    if (this.pending) throw new Error('rust face worker is busy')
    const id = ctx.job.id
    return await new Promise<FaceRustResult>((resolve, reject) => {
      this.pending = { resolve, reject, ctx, startedAt: Date.now(), emitted: 0 }
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
        frames: Number(result.frames ?? 0),
        faces: Number(result.faces ?? 0),
        matches: Number(result.matches ?? 0),
        jsonl: String(result.jsonl ?? ''),
        engine: 'rust',
        device: String(result.device ?? 'cuda:0'),
      })
      this.pending = null
      return
    }
    if (event.type === 'error') {
      const code = typeof event.code === 'string' ? event.code : 'face-failed'
      const message = typeof event.message === 'string' ? event.message : 'rust face worker failed'
      const error = Object.assign(new Error(message), { code })
      pending.reject(error)
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
      /* процесс уже мог завершиться */
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

/** Метка файла для журнала: имя кадра/видео без каталога. */
export function faceJobLabel(path: string): string {
  return basename(path)
}
