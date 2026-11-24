/** Общие типы движка: настройки, факты GPU/доктора, задания, история, аренда. */

export type DeviceKind = 'gpu' | 'cpu'

export interface DeviceRef {
  kind: DeviceKind
  /** Индекс GPU, когда kind === 'gpu'. */
  index?: number
  /** Человекочитаемая метка, например "RTX 3070". */
  name?: string
}

export interface Settings {
  ffmpegPath: string
  ffmpegGpuPath: string
  ffprobePath: string
  ffprobeGpuPath: string
  pythonPath: string
  uvPath: string
  modelsDir: string
  ocrModelDir: string
  /** Область табло по умолчанию для OCR, если вызывающий не задал region. */
  ocrRegion: string
  /** Приоритет OCR: DeepSeek Tools API, затем локальный PP-OCRv5, либо заданный явно. */
  ocrEngine: 'auto' | 'deepseek' | 'local'
  /** Корень DeepSeek API для OCR. */
  deepseekBaseUrl: string
  /** Имя credential-переменной (или переменной окружения) с ключом DeepSeek. */
  deepseekApiKeyEnv: string
  /** Таймаут запроса DeepSeek OCR (мс). */
  deepseekTimeoutMs: number
  /** Бэкенд скана лиц: rust (постоянный воркер), python или авто. */
  faceWorker: 'auto' | 'rust' | 'python'
  /** Шрифт для титров счёта в монтаже. */
  montageFontPath: string
  /** Модель x2-апскейла (SPAN fp16 ONNX). */
  upscaleModelPath: string
  /** Размер тайла и перекрытие для апскейла (CUDA). */
  upscaleBatch: number
  upscaleProvider: 'auto' | 'cuda' | 'tensorrt'
  /** Кодек/качество видео-апскейла (NVENC). */
  upscaleCodec: 'hevc' | 'h264'
  upscaleQuality: 'fast' | 'balanced' | 'quality'
  /** Модель трекинга мяча (ONNX TrackNet/WASB). */
  ballModelPath: string
  ballThreshold: number
  ballMaxGapFrames: number
  ballAnnotate: boolean
  ballDevice: 'auto' | 'cuda' | 'cpu'
  outputRoot: string
  torchIndexUrl: string
  gpuConcurrency: number
  cpuConcurrency: number
  historyLimit: number
  jobHistoryLimit: number
  httpPort: number
  mcpHttpEnabled: boolean
  allowCpuFallback: boolean
  apiToken: string
  /** Дополнительные записи LD_LIBRARY_PATH для GPU-процессов Python. */
  extraLdLibraryPath: string
}

export interface GpuInfo {
  index: number
  uuid?: string
  name: string
  totalMB: number
  usedMB: number
  freeMB: number
  /** Сколько VRAM занято видимыми compute-процессами. */
  usedByProcessesMB?: number
  /** Разница used − процессы: занято вне namespace (другой контейнер/хост). */
  otherMB?: number
  driver: string
  computeCap: string
}

export interface GpuProcess {
  gpuUuid: string
  pid: number
  name: string
  usedMB: number
}

export interface GpuStatus {
  available: boolean
  passthrough: boolean
  gpus: GpuInfo[]
  processes: GpuProcess[]
  error: string | null
  hints: string[]
  checkedAt: number
  source: 'linux' | 'none'
}

export type DoctorLevel = 'ok' | 'warn' | 'fail' | 'unknown'

export interface DoctorCheck {
  id: string
  title: string
  level: DoctorLevel
  detail: string
  /** Готовая команда/рецепт, если проверка требует действия пользователя. */
  fix?: string
  /** Машинное значение для группировки на клиенте. */
  group: 'gpu' | 'ffmpeg' | 'vulkan' | 'onnx' | 'torch' | 'yolo' | 'ocr' | 'python'
}

export interface DoctorReport {
  checkedAt: number
  checks: DoctorCheck[]
  summary: { ok: number; warn: number; fail: number; unknown: number }
}

export interface FfmpegCaps {
  path: string
  version: string
  encoders: string[]
  decoders: string[]
  filters: string[]
  hwaccels: string[]
}

export type JobKind = 'ffmpeg' | 'yolo' | 'onnx' | 'face' | 'ocr' | 'python' | 'setup' | 'upscale' | 'ball'
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'

export interface JobProgress {
  percent: number | null
  phase: string
  message: string
  processedMs: number | null
  totalMs: number | null
  frame: number | null
  totalFrames: number | null
  fps: number | null
  speed: number | null
  bytes: number | null
  segmentsDone: number | null
  segmentsTotal: number | null
  etaMs: number | null
  updatedAt: number
  /** Последняя измеренная скорость прогресса (единицы в секунду, зависит от движка). */
  rate: number | null
}

export interface Who {
  /** tool = подсказка DSH, http = REST-вызов, mcp = внешний MCP-клиент. */
  source: 'tool' | 'http' | 'mcp' | 'system'
  sessionId: string | null
  label: string | null
  client: string | null
}

export interface JobOutput {
  path: string
  bytes: number | null
  durationSec: number | null
}

export interface JobRecord {
  id: string
  kind: JobKind
  operation: string
  status: JobStatus
  who: Who
  device: DeviceRef
  engine: string | null
  argv: string[]
  params: Record<string, unknown>
  progress: JobProgress
  createdAt: number
  startedAt: number | null
  finishedAt: number | null
  durationMs: number | null
  exitCode: number | null
  signal: string | null
  outputs: JobOutput[]
  error: { code: string; message: string; details?: unknown } | null
  logTail: string[]
  logPath: string | null
  hintId: string | null
  /** pid дочерних процессов задания (для watchdog и жёсткой отмены). */
  pids?: number[]
  /** Предупреждения плана (фильтры, фолбэки энкодера). */
  warnings?: string[]
  /** Жив ли ещё процесс running-задания (вычисляется при выдаче). */
  alive?: boolean
  /** Нет прогресса дольше порога (вычисляется при выдаче). */
  stalled?: boolean
}

export interface CallRecord {
  id: string
  tool: string
  operation: string | null
  who: Who
  startedAt: number
  durationMs: number
  ok: boolean
  error: { code: string; message: string } | null
  jobId: string | null
  target: string | null
}

export interface LeaseRecord {
  id: string
  holder: string
  who: Who
  devices: number[]
  createdAt: number
  expiresAt: number
}

export interface HintRecord {
  id: string
  tool: string
  argsHash: string
  who: Who
  at: number
}

export interface OcrCapability {
  engine: 'auto' | 'deepseek' | 'local'
  deepseekKeyConfigured: boolean
  deepseekBaseUrl: string
  apiKeyEnv: string
  localModelDir: string
  region: string
}

export interface FaceCapability {
  mode: 'auto' | 'rust' | 'python'
  rust: { ok: boolean; binary: string; library: string | null; reason: string }
}

export interface ServerState {
  version: string
  uptimeSec: number
  apiBaseUrl: string
  mcpCommand: string
  gpu: GpuStatus
  doctor: DoctorReport
  jobs: JobRecord[]
  queue: { running: number; queued: number; stalled: number; aliveRunning: number }
  leases: LeaseRecord[]
  calls: CallRecord[]
  settings: Omit<Settings, 'apiToken'>
  capabilities?: { ocr: OcrCapability; face: FaceCapability }
  ffmpeg?: FfmpegDiagnostics
}

export interface FfmpegDiagnostics {
  binary: string
  gpu: boolean
  encoders: Record<string, boolean>
  filters: Record<string, boolean>
  hwaccels: string[]
}

export interface JobContext {
  readonly job: JobRecord
  readonly signal: AbortSignal
  readonly device: DeviceRef
  phase(phase: string, message?: string): void
  progress(patch: Partial<JobProgress>): void
  log(line: string): void
  output(entry: JobOutput): void
  /** Регистрирует pid дочернего процесса, чтобы watchdog/отмена могли его найти. */
  trackPid(pid: number): void
}

export type JobExecutor = (context: JobContext) => Promise<void>

export interface EngineError extends Error {
  code?: string
  details?: unknown
}
