/** Мост к Python: проверка интерпретатора и запуск JSONL-воркеров. */
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { existsSync } from 'node:fs'
import type { JobContext, Settings } from './types.js'
import type { PythonProbe } from './doctor.js'
import { gpuEnv } from './settings.js'
import { run, stream } from './proc.js'

/** Корень пакета, вычисленный из расположения собранного модуля (dist/engine). */
export function packageRoot(): string {
  return join(dirname(fileURLToPath(import.meta.url)), '..', '..')
}

export function pythonScript(name: string): string {
  return join(packageRoot(), 'python', name)
}

const PROBE = `
import json, sys
result = {"version": sys.version.split()[0], "torch": None, "torchCuda": False, "torchCudaVersion": None,
          "ort": None, "providers": [], "ultralytics": None, "cv2": None, "numpy": None, "cudnn": None, "error": None, "deviceCount": 0}
try:
    import torch
    result["torch"] = torch.__version__
    result["torchCuda"] = bool(torch.cuda.is_available())
    result["torchCudaVersion"] = getattr(torch.version, "cuda", None)
    try:
        result["deviceCount"] = int(torch.cuda.device_count()) if result["torchCuda"] else 0
    except Exception:
        result["deviceCount"] = 0
except Exception as exc:
    result["error"] = "torch: " + str(exc)
try:
    import onnxruntime as ort
    result["ort"] = ort.__version__
    try:
        result["providers"] = list(ort.get_available_providers())
    except Exception:
        result["providers"] = []
except Exception as exc:
    result["error"] = (result["error"] + "; " if result["error"] else "") + "onnxruntime: " + str(exc)
try:
    import ultralytics
    result["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
except Exception as exc:
    result["error"] = (result["error"] + "; " if result["error"] else "") + "ultralytics: " + str(exc)
try:
    import cv2
    result["cv2"] = cv2.__version__
except Exception as exc:
    result["error"] = (result["error"] + "; " if result["error"] else "") + "cv2: " + str(exc)
try:
    import numpy
    result["numpy"] = numpy.__version__
except Exception as exc:
    result["error"] = (result["error"] + "; " if result["error"] else "") + "numpy: " + str(exc)
try:
    import ctypes
    ctypes.CDLL("libcudnn.so.9")
    result["cudnn"] = "libcudnn.so.9"
except Exception as exc:
    result["cudnn"] = None
    result["error"] = (result["error"] + "; " if result["error"] else "") + "cudnn: " + str(exc)
print(json.dumps(result))
`

/** Запускает интерпретатор и собирает версии и провайдеры. */
export async function pythonProbe(settings: Settings): Promise<PythonProbe | null> {
  if (!existsSync(settings.pythonPath) && settings.pythonPath !== 'python3') return null
  try {
    const result = await run(settings.pythonPath, ['-c', PROBE], {
      timeoutMs: 60_000,
      env: gpuEnv(settings, undefined),
      maxOutput: 2 * 1024 * 1024,
    })
    if (result.code !== 0) return null
    const line = result.stdout.trim().split(/\r?\n/).filter(Boolean).pop()
    if (!line) return null
    return JSON.parse(line) as PythonProbe
  } catch {
    return null
  }
}

export interface WorkerOptions {
  script: string
  config: Record<string, unknown>
  settings: Settings
  ctx: JobContext
  deviceIndex: number | null
  timeoutMs?: number
}

export interface WorkerEvent {
  type: string
  [key: string]: unknown
}

/** Запускает Python-воркер, превращая JSONL в прогресс и лог задания. */
export async function runPythonWorker(options: WorkerOptions): Promise<unknown> {
  const { ctx, settings, script } = options
  const config = JSON.stringify(options.config)
  let result: unknown = null
  const failure: { value: { code: string; message: string; details?: unknown } | null } = { value: null }
  const onLine = (line: string): void => {
    const trimmed = line.trim()
    if (!trimmed.startsWith('{')) {
      ctx.log(trimmed)
      return
    }
    let event: WorkerEvent
    try {
      event = JSON.parse(trimmed) as WorkerEvent
    } catch {
      ctx.log(trimmed)
      return
    }
    if (event.type === 'log') {
      ctx.log(String(event.message ?? ''))
      return
    }
    if (event.type === 'progress') {
      const phase = typeof event.phase === 'string' ? event.phase : ctx.job.progress.phase
      const message = typeof event.message === 'string' ? event.message : ''
      const patch: Record<string, unknown> = { phase }
      if (message) patch.message = message
      for (const key of ['percent', 'frame', 'totalFrames', 'fps', 'etaMs', 'processedMs', 'totalMs', 'rate', 'segmentsDone', 'segmentsTotal']) {
        if (typeof event[key] === 'number' && Number.isFinite(event[key] as number)) patch[key] = event[key]
      }
      if (typeof patch.percent === 'number') {
        patch.percent = Math.max(0, Math.min(100, patch.percent as number))
      }
      ctx.progress(patch)
      return
    }
    if (event.type === 'result') {
      result = event.value ?? null
      return
    }
    if (event.type === 'error') {
      failure.value = {
        code: typeof event.code === 'string' ? event.code : 'worker-failed',
        message: typeof event.message === 'string' ? event.message : 'Python worker failed',
        details: event.details,
      }
      ctx.log(`error: ${failure.value.message}`)
      return
    }
    ctx.log(trimmed)
  }
  const outcome = await stream(settings.pythonPath, [script, '--config', config], {
    env: gpuEnv(settings, options.deviceIndex ?? undefined),
    signal: ctx.signal,
    onStdoutLine: onLine,
    onStderrLine: (line) => ctx.log(line),
    onSpawn: (pid) => ctx.trackPid(pid),
    tailLines: 300,
    timeoutMs: options.timeoutMs,
  })
  if (failure.value) throw Object.assign(new Error(failure.value.message), { code: failure.value.code, details: failure.value.details })
  if (outcome.code !== 0) {
    const tail = outcome.tail.slice(-6).join(' | ')
    throw Object.assign(new Error(`Python worker exited ${outcome.code}${tail ? `: ${tail}` : ''}`), { code: 'worker-failed' })
  }
  return result
}
