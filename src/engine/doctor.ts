/** Доктор возможностей: всё, что блокирует GPU-этап, с рецептами исправления. */
import { existsSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import type { DoctorCheck, DoctorReport, FfmpegCaps, GpuStatus, Settings } from './types.js'
import { probeFfmpeg } from './ffmpeg.js'
import { run } from './proc.js'
import { gpuEnv } from './settings.js'
import { upscaleAvailable } from './upscale-rust.js'
import { ballAvailable } from './ball-rust.js'

export interface DoctorDeps {
  settings: () => Settings
  gpu: (refresh?: boolean) => Promise<GpuStatus>
  pythonProbe: () => Promise<PythonProbe | null>
}

export interface PythonProbe {
  version: string
  torch: string | null
  torchCuda: boolean
  torchCudaVersion: string | null
  ort: string | null
  providers: string[]
  ultralytics: string | null
  cv2: string | null
  numpy: string | null
  cudnn: string | null
  error: string | null
  deviceCount?: number
}

const PYTHON_PROBE = `
import json, sys
result = {"version": sys.version.split()[0], "torch": None, "torchCuda": False, "torchCudaVersion": None,
          "ort": None, "providers": [], "ultralytics": None, "cv2": None, "numpy": None, "cudnn": None, "error": None}
try:
    import torch
    result["torch"] = torch.__version__
    result["torchCuda"] = bool(torch.cuda.is_available())
    result["torchCudaVersion"] = getattr(torch.version, "cuda", None)
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

export class Doctor {
  private vulkanCache: { available: boolean; hardware: boolean; detail: string; at: number } | null = null
  private upscaleCache: { ok: boolean; detail: string; at: number } | null = null
  private cached: DoctorReport | null = null
  private cachedAt = 0
  private ffmpegCache = new Map<string, FfmpegCaps>()

  constructor(private readonly deps: DoctorDeps) {}

  /** Проверяет бинари ffmpeg с коротким кешем. */
  async caps(path: string): Promise<FfmpegCaps> {
    const cached = this.ffmpegCache.get(path)
    if (cached) return cached
    const caps = await probeFfmpeg(path)
    this.ffmpegCache.set(path, caps)
    return caps
  }

  invalidate(): void {
    this.cached = null
    this.ffmpegCache.clear()
  }

  async probePython(): Promise<PythonProbe | null> {
    return await this.deps.pythonProbe()
  }

  /** Выполняет проверки. Кеш 30 секунд, если не запрошено обновление. */
  async report(refresh = false): Promise<DoctorReport> {
    if (!refresh && this.cached && Date.now() - this.cachedAt < 30_000) return this.cached
    const settings = this.deps.settings()
    const checks: DoctorCheck[] = []
    const [gpu, python, ffmpeg, ffmpegGpu] = await Promise.all([
      this.deps.gpu(refresh),
      this.deps.pythonProbe().catch(() => null),
      this.caps(settings.ffmpegPath),
      this.caps(settings.ffmpegGpuPath),
    ])

    const nodes = existsSync('/dev/dxg') || existsSync('/dev/nvidia0')
    const byId = (id: string): DoctorCheck | undefined => checks.find((check) => check.id === id)

    if (gpu.available) {
      checks.push({
        id: 'gpu-nvidia-smi',
        title: 'NVIDIA GPUs',
        level: 'ok',
        detail: gpu.gpus.map((entry) => `#${entry.index} ${entry.name} ${Math.round(entry.freeMB / 1024)} GB free / ${Math.round(entry.totalMB / 1024)} GB`).join('; '),
        group: 'gpu',
      })
    } else {
      checks.push({
        id: 'gpu-nvidia-smi',
        title: 'NVIDIA GPUs',
        level: 'fail',
        detail: gpu.error ?? 'nvidia-smi is unavailable',
        fix: 'Run nvidia-smi in the WSL terminal that starts DSH; update the Windows driver and `wsl --shutdown` if NVML fails.',
        group: 'gpu',
      })
    }
    checks.push({
      id: 'gpu-device-nodes',
      title: 'GPU device nodes',
      level: nodes ? 'ok' : 'fail',
      detail: nodes ? '/dev/dxg or /dev/nvidia0 is visible to the service process' : '/dev/dxg is missing in this process namespace',
      fix: 'Start DSH in the WSL shell where /dev/dxg exists; the agent bash sandbox cannot see GPU devices.',
      group: 'gpu',
    })
    if (gpu.processes.length) {
      checks.push({
        id: 'gpu-processes',
        title: 'External GPU processes',
        level: 'warn',
        detail: gpu.processes.map((entry) => `pid ${entry.pid} ${entry.name} ${entry.usedMB} MiB`).join('; '),
        group: 'gpu',
      })
    }

    const nvenc = ['h264_nvenc', 'hevc_nvenc', 'av1_nvenc'].filter((name) => ffmpegGpu.encoders.includes(name))
    const cuvid = ffmpegGpu.decoders.filter((name) => name.endsWith('_cuvid'))
    checks.push({
      id: 'ffmpeg-gpu',
      title: 'ffmpeg GPU build',
      level: ffmpegGpu.version ? (nvenc.length && cuvid.length ? 'ok' : 'warn') : 'fail',
      detail: ffmpegGpu.version
        ? `${ffmpegGpu.path}: ${nvenc.length} nvenc, ${cuvid.length} cuvid, ${ffmpegGpu.filters.includes('scale_cuda') ? 'scale_cuda' : 'no scale_cuda'}`
        : `${ffmpegGpu.path} is missing`,
      fix: 'Point Settings → ffmpeg GPU binary to a build with NVENC and CUVID support.',
      group: 'ffmpeg',
    })
    checks.push({
      id: 'ffmpeg-cpu',
      title: 'ffmpeg CPU fallback',
      level: ffmpeg.version ? 'ok' : 'fail',
      detail: ffmpeg.version ? `${ffmpeg.path}: ${ffmpeg.version}` : `${ffmpeg.path} is missing`,
      group: 'ffmpeg',
    })

    const montageFilters = ['concat', 'drawtext', 'overlay', 'subtitles', 'amix', 'adelay', 'minterpolate', 'libplacebo', 'fruc_vulkan']
    const missingMontage = montageFilters.filter((name) => !ffmpegGpu.filters.includes(name))
    checks.push({
      id: 'montage-filters',
      title: 'Montage filters',
      level: missingMontage.length === 0 ? 'ok' : missingMontage.includes('concat') || missingMontage.includes('drawtext') ? 'fail' : 'warn',
      detail: missingMontage.length
        ? `missing: ${missingMontage.join(', ')}`
        : 'concat, drawtext, overlay, subtitles, amix, adelay, minterpolate, libplacebo, fruc_vulkan',
      fix: 'Use the ffmpeg-nvenc build in Settings; concat, drawtext, overlay and subtitles are needed for montage.',
      group: 'ffmpeg',
    })
    checks.push({
      id: 'montage-font',
      title: 'Montage score font',
      level: settings.montageFontPath && existsSync(settings.montageFontPath) ? 'ok' : 'warn',
      detail: settings.montageFontPath && existsSync(settings.montageFontPath) ? settings.montageFontPath : 'font file is not set; score events will fail',
      fix: 'Set Settings → Montage font to a .ttf/.otf file (e.g. /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf).',
      group: 'ffmpeg',
    })
    const vulkan = await this.vulkanCapability(settings)
    const upscale = upscaleAvailable(settings)
    checks.push({
      id: 'upscale-worker',
      title: 'x2 upscale worker (SPAN fp16)',
      level: existsSync(upscale.binary) ? 'ok' : 'fail',
      detail: existsSync(upscale.binary) ? upscale.binary : `missing: ${upscale.binary} (npm run build:rust)`,
      fix: 'Build the Rust worker with npm run build:rust.',
      group: 'ffmpeg',
    })
    checks.push({
      id: 'upscale-model',
      title: 'x2 upscale model (SPAN fp16)',
      level: existsSync(upscale.model) ? 'ok' : 'warn',
      detail: existsSync(upscale.model) ? upscale.model : `not found: ${upscale.model} (set Settings → upscaleModelPath)`,
      fix: 'Export span2x_ch48_fp16.onnx and set upscaleModelPath (or GPUTOOL_UPSCALE_MODEL).',
      group: 'onnx',
    })
    const ball = ballAvailable(settings)
    checks.push({
      id: 'ball-worker',
      title: 'Ball tracking worker',
      level: existsSync(ball.binary) ? 'ok' : 'fail',
      detail: existsSync(ball.binary) ? ball.binary : `missing: ${ball.binary} (npm run build:rust)`,
      fix: 'Build the Rust workers with npm run build:rust.',
      group: 'ffmpeg',
    })
    checks.push({
      id: 'ball-model',
      title: 'Ball model (ONNX)',
      level: existsSync(ball.model) ? 'ok' : 'warn',
      detail: existsSync(ball.model) ? ball.model : `not found: ${ball.model} (set Settings → ballModelPath)`,
      fix: 'Export TrackNetV2/V3 or WASB to ONNX and set ballModelPath (or GPUTOOL_BALL_MODEL).',
      group: 'onnx',
    })

    if (upscale.ok) {
      const probe = await this.upscaleProbe(settings)
      checks.push({
        id: 'upscale-cuda',
        title: 'SPAN x2 fp16 on CUDA',
        level: probe.ok ? 'ok' : 'fail',
        detail: probe.detail,
        fix: probe.ok ? undefined : 'The model must fit entirely on CUDA: check the ORT error, driver and CUDA EP build.',
        group: 'gpu',
      })
    }

    checks.push({
      id: 'montage-vulkan',
      title: 'Montage Vulkan (EWA, fruc)',
      level: vulkan.hardware ? 'ok' : 'warn',
      detail: vulkan.hardware
        ? `${vulkan.detail}: EWA и fruc_vulkan пойдут на GPU`
        : `${vulkan.detail}: EWA идёт на CPU (llvmpipe), fruc заменяется minterpolate`,
      fix: 'For GPU EWA/interpolation install an NVIDIA Vulkan ICD (driver components) and restart WSL.',
      group: 'vulkan',
    })

    if (!python) {
      checks.push({
        id: 'python',
        title: 'Python runtime',
        level: 'fail',
        detail: `Cannot run ${settings.pythonPath}`,
        fix: 'Run Setup → install-runtime in the panel, or point Settings → Python to a working interpreter.',
        group: 'python',
      })
    } else {
      checks.push({
        id: 'python',
        title: 'Python runtime',
        level: 'ok',
        detail: `${settings.pythonPath} (${python.version})`,
        group: 'python',
      })
      checks.push({
        id: 'onnx-runtime',
        title: 'ONNX Runtime CUDA',
        level: python.ort && python.providers.includes('CUDAExecutionProvider') ? 'ok' : 'fail',
        detail: python.ort
          ? `onnxruntime ${python.ort}: ${python.providers.join(', ') || 'no providers'}`
          : 'onnxruntime is not installed',
        fix: 'Setup → install-runtime installs onnxruntime-gpu; cuDNN is provided by the CUDA torch wheel.',
        group: 'onnx',
      })
      checks.push({
        id: 'cudnn',
        title: 'cuDNN',
        level: python.cudnn ? 'ok' : 'fail',
        detail: python.cudnn ? `loaded ${python.cudnn}` : 'libcudnn.so.9 is not loadable',
        fix: 'Setup → install-runtime installs torch with the cu126 CUDA wheels (they bring cuDNN).',
        group: 'onnx',
      })
      checks.push({
        id: 'torch-cuda',
        title: 'torch CUDA',
        level: python.torchCuda ? 'ok' : python.torch ? 'warn' : 'fail',
        detail: python.torch
          ? `torch ${python.torch}${python.torchCudaVersion ? ` (CUDA ${python.torchCudaVersion})` : ''}; cuda available=${python.torchCuda}${python.torchCuda && python.deviceCount ? `, устройств: ${python.deviceCount}` : ''}`
          : 'torch is not installed',
        fix: python.torchCuda ? undefined : 'Setup → install-runtime installs torch cu126 into the plugin venv.',
        group: 'torch',
      })
      checks.push({
        id: 'ultralytics',
        title: 'ultralytics (YOLO11)',
        level: python.ultralytics ? 'ok' : 'fail',
        detail: python.ultralytics ? `ultralytics ${python.ultralytics}` : 'ultralytics is not installed',
        fix: 'Setup → install-runtime installs ultralytics.',
        group: 'yolo',
      })
      checks.push({
        id: 'opencv',
        title: 'OpenCV',
        level: python.cv2 ? 'ok' : 'fail',
        detail: python.cv2 ? `opencv ${python.cv2}` : 'cv2 is not installed',
        fix: 'Setup → install-runtime installs opencv-python-headless.',
        group: 'python',
      })
      if (python.error && !byId('python-error')) {
        checks.push({
          id: 'python-error',
          title: 'Python import diagnostics',
          level: 'warn',
          detail: python.error,
          group: 'python',
        })
      }
    }

    const yoloModel = join(settings.modelsDir, 'yolo11n.pt')
    checks.push({
      id: 'yolo-model',
      title: 'YOLO11 weights',
      level: existsSync(yoloModel) ? 'ok' : 'warn',
      detail: existsSync(yoloModel) ? yoloModel : `${yoloModel} is missing; ultralytics may download it on first use`,
      group: 'yolo',
    })
    const det = join(settings.ocrModelDir, 'PP-OCRv5_mobile_det.onnx')
    const rec = join(settings.ocrModelDir, 'PP-OCRv5_mobile_rec.onnx')
    checks.push({
      id: 'ocr-models',
      title: 'PP-OCRv5 models',
      level: existsSync(det) && existsSync(rec) ? 'ok' : 'warn',
      detail: existsSync(det) && existsSync(rec) ? settings.ocrModelDir : 'PP-OCRv5 det/rec ONNX models are missing',
      fix: 'Set Settings → OCR model directory to the folder with PP-OCRv5_mobile_det.onnx / _rec.onnx.',
      group: 'ocr',
    })
    for (const name of ['yunet.onnx', 'sface.onnx']) {
      const path = join(settings.modelsDir, name)
      checks.push({
        id: `face-${name}`,
        title: `Face model ${name}`,
        level: existsSync(path) ? 'ok' : 'fail',
        detail: existsSync(path) ? path : `${path} is missing`,
        fix: 'Point Settings → models directory to the folder with yunet.onnx and sface.onnx.',
        group: 'onnx',
      })
    }

    const summary = { ok: 0, warn: 0, fail: 0, unknown: 0 }
    for (const check of checks) summary[check.level] += 1
    const report: DoctorReport = { checkedAt: Date.now(), checks, summary }
    this.cached = report
    this.cachedAt = Date.now()
    return report
  }


  /**
   * Возможности Vulkan для монтажа: аппаратный ICD или программный llvmpipe.
   * Результат кэшируется на минуту — проба запускает ffmpeg.
   */
  async vulkanCapability(settings: Settings): Promise<{ available: boolean; hardware: boolean; detail: string }> {
    const fresh = this.vulkanCache && Date.now() - this.vulkanCache.at < 60_000
    if (fresh && this.vulkanCache) return this.vulkanCache
    const path = settings.ffmpegGpuPath
    const fallback = { available: false, hardware: false, detail: `${path} is missing`, at: Date.now() }
    if (!existsSync(path)) {
      this.vulkanCache = fallback
      return fallback
    }
    const result = await run(
      path,
      ['-hide_banner', '-loglevel', 'info', '-init_hw_device', 'vulkan=vk', '-f', 'lavfi', '-i', 'color=black:s=64x64:d=0.05', '-frames:v', '1', '-f', 'null', '-'],
      { timeoutMs: 30_000, env: gpuEnv(settings, undefined), maxOutput: 1_048_576 },
    )
    const output = `${result.stdout}\n${result.stderr}`
    const software = /llvmpipe|swiftshader|software rasterizer/i.test(output)
    const selected = /(Using device:[^\n]*|Device \d+ selected:[^\n]*)/.exec(output)?.[0]?.replace(/\s+/g, ' ').trim()
    const value = {
      available: result.code === 0,
      hardware: result.code === 0 && !software,
      detail: result.code !== 0 ? 'Vulkan init failed' : software ? `software Vulkan (${selected ?? 'llvmpipe'})` : selected ?? 'Vulkan device created',
      at: Date.now(),
    }
    this.vulkanCache = value
    return value
  }

  /**
   * Пробное создание CUDA-сессии воркера с `disable_cpu_ep_fallback`.
   * Дорогая операция (загрузка модели), поэтому кэшируется на 5 минут.
   */
  async upscaleProbe(settings: Settings): Promise<{ ok: boolean; detail: string }> {
    if (this.upscaleCache && Date.now() - this.upscaleCache.at < 300_000) {
      return { ok: this.upscaleCache.ok, detail: this.upscaleCache.detail }
    }
    const availability = upscaleAvailable(settings)
    if (!availability.ok || !availability.library) {
      const detail = availability.reason
      this.upscaleCache = { ok: false, detail, at: Date.now() }
      return { ok: false, detail }
    }
    const child = await run(
      availability.binary,
      ['--probe', availability.model],
      {
        timeoutMs: 240_000,
        env: { ...gpuEnv(settings, undefined), ORT_DYLIB_PATH: availability.library },
        maxOutput: 2 * 1024 * 1024,
      },
    )
    const line = child.stdout.split(/\r?\n/).reverse().find((entry) => entry.includes('"probe"'))
    let detail = `probe exited ${child.code}`
    let ok = false
    if (line) {
      try {
        const parsed = JSON.parse(line) as { type?: string; scale?: number; precision?: string; loadMs?: number; msPerFrame64?: number }
        ok = parsed.type === 'probe' && parsed.scale === 2
        detail = ok
          ? `CUDA ok: x${parsed.scale} ${parsed.precision}, load ${parsed.loadMs ?? '?'} ms, infer ${parsed.msPerFrame64 ?? '?'} ms (64x64)`
          : `CUDA probe failed: ${line.slice(0, 300)}`
      } catch {
        detail = `CUDA probe returned non-JSON: ${line.slice(0, 200)}`
      }
    } else if (child.stderr) {
      detail = `CUDA probe failed: ${child.stderr.trim().split(/\r?\n/).slice(-2).join(' | ').slice(0, 300)}`
    }
    this.upscaleCache = { ok, detail, at: Date.now() }
    return { ok, detail }
  }
}
