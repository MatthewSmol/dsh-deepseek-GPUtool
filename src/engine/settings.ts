/** Настройки: определённые умолчания для этой машины и сохранённые переопределения. */
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { files, gputoolHome, newToken, writeJsonAtomic } from './paths.js'
import type { Settings } from './types.js'

/** Известные пути на этой машине; отсутствующие просто пропускаются. */
const KNOWN = {
  ffmpegGpu: '/home/main/test/bin/ffmpeg-nvenc',
  ffprobeGpu: '/home/main/test/bin/ffprobe-nvenc',
  uv: '/home/main/test/.uv/uv',
  models: '/home/main/test/p4/work/models',
  ocr: '/home/main/test/ocr-ru/models-ru',
  legacyPython: '/home/main/test/.venv/cv312/bin/python',
}

/** Лучший существующий ffmpeg для GPU-работы либо системный. */
function pickGpuBinary(path: string, fallback: string): string {
  return existsSync(path) ? path : fallback
}

export function defaultSettings(): Settings {
  const python = existsSync(files.python()) ? files.python() : existsSync(KNOWN.legacyPython) ? KNOWN.legacyPython : 'python3'
  return {
    ffmpegPath: '/usr/bin/ffmpeg',
    ffmpegGpuPath: pickGpuBinary(KNOWN.ffmpegGpu, '/usr/bin/ffmpeg'),
    ffprobePath: '/usr/bin/ffprobe',
    ffprobeGpuPath: pickGpuBinary(KNOWN.ffprobeGpu, '/usr/bin/ffprobe'),
    pythonPath: python,
    uvPath: existsSync(KNOWN.uv) ? KNOWN.uv : 'uv',
    modelsDir: existsSync(KNOWN.models) ? KNOWN.models : join(gputoolHome(), 'models'),
    ocrModelDir: existsSync(KNOWN.ocr) ? KNOWN.ocr : join(gputoolHome(), 'ocr-models'),
    ocrRegion: '',
    // Приоритет «DeepSeek → локальный PP-OCRv5»; сам ключ живёт в credential-store DSH.
    ocrEngine: 'auto',
    deepseekBaseUrl: 'https://api.deepseek.com',
    deepseekApiKeyEnv: 'DEEPSEEK_API_KEY',
    deepseekTimeoutMs: 170_000,
    outputRoot: files.outputs(),
    torchIndexUrl: 'https://download.pytorch.org/whl/cu126',
    gpuConcurrency: 1,
    cpuConcurrency: 2,
    historyLimit: 1000,
    jobHistoryLimit: 300,
    httpPort: 0,
    mcpHttpEnabled: false,
    allowCpuFallback: true,
    apiToken: newToken(),
    extraLdLibraryPath: '',
  }
}

const envOverrides = (): Partial<Settings> => {
  const patch: Partial<Settings> = {}
  if (process.env.GPUTOOL_FFMPEG) patch.ffmpegPath = process.env.GPUTOOL_FFMPEG
  if (process.env.GPUTOOL_FFMPEG_GPU) patch.ffmpegGpuPath = process.env.GPUTOOL_FFMPEG_GPU
  if (process.env.GPUTOOL_FFPROBE) patch.ffprobePath = process.env.GPUTOOL_FFPROBE
  if (process.env.GPUTOOL_FFPROBE_GPU) patch.ffprobeGpuPath = process.env.GPUTOOL_FFPROBE_GPU
  if (process.env.GPUTOOL_PYTHON) patch.pythonPath = process.env.GPUTOOL_PYTHON
  if (process.env.GPUTOOL_UV) patch.uvPath = process.env.GPUTOOL_UV
  if (process.env.GPUTOOL_MODELS_DIR) patch.modelsDir = process.env.GPUTOOL_MODELS_DIR
  if (process.env.GPUTOOL_OCR_MODEL_DIR) patch.ocrModelDir = process.env.GPUTOOL_OCR_MODEL_DIR
  if (process.env.GPUTOOL_OCR_ENGINE === 'deepseek' || process.env.GPUTOOL_OCR_ENGINE === 'local' || process.env.GPUTOOL_OCR_ENGINE === 'auto') {
    patch.ocrEngine = process.env.GPUTOOL_OCR_ENGINE
  }
  if (process.env.GPUTOOL_DEEPSEEK_BASE_URL) patch.deepseekBaseUrl = process.env.GPUTOOL_DEEPSEEK_BASE_URL
  if (process.env.GPUTOOL_DEEPSEEK_API_KEY_ENV) patch.deepseekApiKeyEnv = process.env.GPUTOOL_DEEPSEEK_API_KEY_ENV
  if (process.env.GPUTOOL_OUTPUT_ROOT) patch.outputRoot = process.env.GPUTOOL_OUTPUT_ROOT
  if (process.env.GPUTOOL_HTTP_PORT) patch.httpPort = Number(process.env.GPUTOOL_HTTP_PORT) || 0
  if (process.env.GPUTOOL_API_TOKEN) patch.apiToken = process.env.GPUTOOL_API_TOKEN
  if (process.env.GPUTOOL_TORCH_INDEX) patch.torchIndexUrl = process.env.GPUTOOL_TORCH_INDEX
  return patch
}

function sanitize(input: unknown, base: Settings): Settings {
  const value = (input && typeof input === 'object' ? input : {}) as Partial<Settings>
  const merged: Settings = { ...base, ...envOverrides(), ...value }
  const clampInt = (n: unknown, min: number, max: number, fallback: number): number => {
    const parsed = typeof n === 'number' ? Math.round(n) : Number(n)
    if (!Number.isFinite(parsed)) return fallback
    return Math.min(max, Math.max(min, parsed))
  }
  merged.gpuConcurrency = clampInt(merged.gpuConcurrency, 1, 8, 1)
  merged.cpuConcurrency = clampInt(merged.cpuConcurrency, 1, 8, 2)
  merged.historyLimit = clampInt(merged.historyLimit, 50, 5000, 1000)
  merged.jobHistoryLimit = clampInt(merged.jobHistoryLimit, 20, 2000, 300)
  merged.httpPort = clampInt(merged.httpPort, 0, 65535, 0)
  merged.allowCpuFallback = merged.allowCpuFallback !== false
  if (merged.ocrEngine !== 'deepseek' && merged.ocrEngine !== 'local') merged.ocrEngine = 'auto'
  merged.deepseekTimeoutMs = clampInt(merged.deepseekTimeoutMs, 5_000, 600_000, 170_000)
  merged.mcpHttpEnabled = merged.mcpHttpEnabled === true
  if (!merged.apiToken || typeof merged.apiToken !== 'string') merged.apiToken = newToken()
  return merged
}

export class SettingsStore {
  private current: Settings | null = null

  get(): Settings {
    if (this.current) return this.current
    let stored: unknown = null
    try {
      if (existsSync(files.settings())) stored = JSON.parse(readFileSync(files.settings(), 'utf8'))
    } catch {
      stored = null
    }
    this.current = sanitize(stored, defaultSettings())
    return this.current
  }

  async persist(settings: Settings): Promise<Settings> {
    const next = sanitize(settings, defaultSettings())
    this.current = next
    await writeJsonAtomic(files.settings(), next)
    return next
  }

  async update(patch: Partial<Settings>): Promise<Settings> {
    const next = sanitize({ ...this.get(), ...patch }, defaultSettings())
    return await this.persist(next)
  }

  /** Публичное представление без секретов. */
  view(): Omit<Settings, 'apiToken'> {
    const { apiToken: _ignored, ...rest } = this.get()
    return rest
  }
}

/** Каталог site-packages venv: первый найденный lib/python-версия/site-packages. */
function venvSitePackages(venvRoot: string): string | null {
  const libRoot = join(venvRoot, 'lib')
  try {
    for (const entry of readdirSync(libRoot)) {
      if (!entry.startsWith('python')) continue
      const candidate = join(libRoot, entry, 'site-packages')
      if (existsSync(candidate)) return candidate
    }
  } catch {
    /* venv без lib — библиотеки не подключаем */
  }
  return null
}

/** Окружение GPU-процессов: выбор устройства и пути к библиотекам CUDA. */
export function gpuEnv(settings: Settings, deviceIndex: number | undefined): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env }
  if (deviceIndex !== undefined) env.CUDA_VISIBLE_DEVICES = String(deviceIndex)
  const paths: string[] = []
  if (settings.extraLdLibraryPath) paths.push(settings.extraLdLibraryPath)
  const python = settings.pythonPath
  const venvRoot = python.includes('/bin/python') ? python.slice(0, python.indexOf('/bin/python')) : ''
  if (venvRoot) {
    // site-packages берётся через glob интерпретатора: версия Python не угадывается.
    const siteDir = venvSitePackages(venvRoot)
    if (siteDir) {
      const nvidiaLibs = join(siteDir, 'nvidia')
      for (const candidate of [join(siteDir, 'torch', 'lib'), join(nvidiaLibs, 'cudnn', 'lib')]) {
        if (existsSync(candidate)) paths.push(candidate)
      }
      // Пакеты nvidia-*-cu12/cu13 кладут библиотеки в nvidia/<пакет>/lib.
      try {
        for (const pkg of readdirSync(nvidiaLibs)) {
          const lib = join(nvidiaLibs, pkg, 'lib')
          if (existsSync(lib)) paths.push(lib)
        }
      } catch {
        /* каталога nvidia нет — это нормально для CPU-окружения */
      }
    }
  }
  if (process.env.LD_LIBRARY_PATH) paths.push(process.env.LD_LIBRARY_PATH)
  if (paths.length) env.LD_LIBRARY_PATH = paths.join(':')
  env.PYTHONUNBUFFERED = '1'
  // PREINSTALLED: ultralytics не пытается ставить пакеты через pip внутри воркера.
  env.YOLO_AUTOINSTALL = 'false'
  env.YOLO_CONFIG_DIR = join(gputoolHome(), 'ultralytics')
  env.MPLCONFIGDIR = join(gputoolHome(), 'matplotlib')
  return env
}
