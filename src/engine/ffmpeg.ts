/** Проверка возможностей ffmpeg, сборка аргументов и разбор прогресса. */
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { mkdir, writeFile } from 'node:fs/promises'
import type { FfmpegCaps, Settings } from './types.js'
import { run } from './proc.js'
import { gpuEnv } from './settings.js'

export interface MontageScene {
  input: string
  startSec?: number
  durationSec?: number
  endSec?: number
}

export interface MontageScoreEvent {
  atSec: number
  untilSec?: number
  text: string
}

export interface MontageScore {
  events: MontageScoreEvent[]
  fontFile?: string
  fontSize?: number
  color?: string
  box?: boolean
  boxColor?: string
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center'
  margin?: number
}

export interface MontageAudioTrack {
  path: string
  atSec?: number
  gainDb?: number
  fadeInSec?: number
  fadeOutSec?: number
}

export interface MontageOverlay {
  path: string
  x?: number
  y?: number
  width?: number
  height?: number
  opacity?: number
  fromSec?: number
  toSec?: number
  loop?: boolean
}

export interface MontageSubtitles {
  path: string
  fontsDir?: string
  forceStyle?: string
  kind?: 'subtitles' | 'ass'
}

export interface MontageAudio {
  sceneAudio?: 'keep' | 'drop'
  tracks?: MontageAudioTrack[]
  normalize?: boolean
  bitrateKbps?: number
}

export interface MontageSpec {
  scenes: MontageScene[]
  normalize?: { width: number; height: number }
  upscale?: { width: number; height: number; method?: 'auto' | 'ewa' | 'lanczos' | 'bicubic' }
  fps?: number
  interpolate?: 'auto' | 'fruc' | 'minterpolate' | 'duplicate' | 'off'
  score?: MontageScore
  subtitles?: MontageSubtitles
  overlays?: MontageOverlay[]
  audio?: MontageAudio
  profile?: 'best' | 'fast'
  strict?: boolean
}

export interface FfmpegParams {
  operation: 'encode' | 'decode' | 'remux' | 'trim' | 'split' | 'concat' | 'frames' | 'thumb' | 'raw' | 'montage'
  montage?: MontageSpec
  input?: string
  inputs?: string[]
  output?: string
  outputDir?: string
  codec?: 'h264' | 'hevc' | 'av1' | 'copy'
  quality?: 'fast' | 'balanced' | 'quality'
  /** Переопределяет CQ (NVENC) / CRF (CPU) из пресета quality. */
  cq?: number
  scale?: string
  fps?: number
  audio?: 'copy' | 'aac' | 'none'
  startSec?: number
  endSec?: number
  durationSec?: number
  segmentSec?: number
  segmentCount?: number
  format?: 'png' | 'jpg'
  everySec?: number
  overwrite?: boolean
  device?: 'auto' | number | 'cpu'
  hwaccel?: 'off' | 'auto' | 'cuvid'
  args?: string[]
  progress?: boolean
}

export interface ProbeInfo {
  durationSec: number | null
  format: string | null
  sizeBytes: number | null
  video: { codec: string | null; width: number | null; height: number | null; fps: number | null } | null
  audio: { codec: string | null; channels: number | null } | null
}

/** Читает JSON ffprobe для медиафайла и нормализует его. */
export async function probeMedia(path: string, settings: Settings): Promise<ProbeInfo> {
  if (!existsSync(path)) throw Object.assign(new Error(`Input not found: ${path}`), { code: 'input-missing' })
  const result = await run(
    settings.ffprobePath,
    ['-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', path],
    { timeoutMs: 60_000, maxOutput: 4 * 1024 * 1024 },
  )
  if (result.code !== 0) {
    throw Object.assign(new Error((result.stderr || 'ffprobe failed').trim()), { code: 'probe-failed' })
  }
  let parsed: any
  try {
    parsed = JSON.parse(result.stdout)
  } catch {
    throw Object.assign(new Error('ffprobe returned invalid JSON'), { code: 'probe-failed' })
  }
  const streams: any[] = Array.isArray(parsed.streams) ? parsed.streams : []
  const video = streams.find((stream) => stream.codec_type === 'video')
  const audio = streams.find((stream) => stream.codec_type === 'audio')
  const duration = Number(parsed.format?.duration ?? video?.duration)
  let fps: number | null = null
  if (typeof video?.avg_frame_rate === 'string' && video.avg_frame_rate.includes('/')) {
    const [num, den] = video.avg_frame_rate.split('/').map(Number)
    if (Number.isFinite(num) && Number.isFinite(den) && den > 0) fps = num / den
  }
  return {
    durationSec: Number.isFinite(duration) ? duration : null,
    format: parsed.format?.format_name ?? null,
    sizeBytes: Number.isFinite(Number(parsed.format?.size)) ? Number(parsed.format.size) : null,
    video: video
      ? {
          codec: video.codec_name ?? null,
          width: Number(video.width) || null,
          height: Number(video.height) || null,
          fps,
        }
      : null,
    audio: audio ? { codec: audio.codec_name ?? null, channels: Number(audio.channels) || null } : null,
  }
}

/** Превращает вывод списков ffmpeg в короткие имена. */
export function parseListing(text: string, kind: 'encoders' | 'decoders' | 'filters'): string[] {
  const names: string[] = []
  for (const line of String(text || '').split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    if (kind === 'encoders' || kind === 'decoders') {
      // " V....D h264_nvenc  NVIDIA NVENC H.264 encoder"
      const match = /^[A-Z.]{6}\s+(\S+)/.exec(trimmed)
      if (match) names.push(match[1])
    } else {
      // " TS cas   V->V  Contrast Adaptive Sharpen."
      const match = /^([A-Z.]{2,3})\s+(\S+)\s+[A-Z|]+->[A-Z|]+/.exec(trimmed)
      if (match) names.push(match[2])
    }
  }
  return names
}


/** Последняя длительность из сырых аргументов: -t, иначе -to с учётом -ss. */
export function rawDurationFromArgs(args: string[]): number | null {
  const numberAfter = (flag: string): number | null => {
    let value: number | null = null
    for (let index = 0; index < args.length; index += 1) {
      if (args[index] !== flag) continue
      const parsed = Number(args[index + 1])
      if (Number.isFinite(parsed) && parsed > 0) value = parsed
    }
    return value
  }
  const limit = numberAfter('-t')
  if (limit != null) return limit
  const until = numberAfter('-to')
  if (until == null) return null
  const start = numberAfter('-ss') ?? 0
  return Math.max(0.1, until - start)
}

/** Эвристика выходного файла для raw: последний не-опционный аргумент с точкой. */
export function inferRawOutput(args: string[]): string | null {
  for (let index = args.length - 1; index >= 0; index -= 1) {
    const token = args[index] ?? ''
    if (!token || token.startsWith('-')) continue
    if (/^[0-9.]+$/.test(token)) continue
    if (/^pipe:/.test(token) || token === '/dev/null') continue
    if (token.includes('.')) return token
  }
  return null
}

/** Проверяет бинарь ffmpeg: версия, кодировщики, декодеры, фильтры, hwaccels. */
export async function probeFfmpeg(path: string): Promise<FfmpegCaps> {
  if (!existsSync(path)) {
    return { path, version: '', encoders: [], decoders: [], filters: [], hwaccels: [] }
  }
  const [version, encoders, decoders, filters, hwaccels] = await Promise.all([
    run(path, ['-hide_banner', '-version'], { timeoutMs: 15_000 }),
    run(path, ['-hide_banner', '-encoders'], { timeoutMs: 15_000 }),
    run(path, ['-hide_banner', '-decoders'], { timeoutMs: 15_000 }),
    run(path, ['-hide_banner', '-filters'], { timeoutMs: 15_000 }),
    run(path, ['-hide_banner', '-hwaccels'], { timeoutMs: 15_000 }),
  ])
  const versionLine = version.stdout.split(/\r?\n/)[0] ?? ''
  return {
    path,
    version: versionLine.trim(),
    encoders: parseListing(encoders.stdout, 'encoders'),
    decoders: parseListing(decoders.stdout, 'decoders'),
    filters: parseListing(filters.stdout, 'filters'),
    hwaccels: hwaccels.stdout
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith('Hardware acceleration')),
  }
}

const NVENC_BY_CODEC: Record<string, string> = { h264: 'h264_nvenc', hevc: 'hevc_nvenc', av1: 'av1_nvenc' }
const CPU_BY_CODEC: Record<string, string> = { h264: 'libx264', hevc: 'libx265', av1: 'libsvtav1' }
const QUALITY: Record<string, { nvenc: string[]; cpu: string[] }> = {
  fast: { nvenc: ['-preset', 'p3', '-tune', 'll', '-rc', 'vbr', '-cq', '27'], cpu: ['-preset', 'veryfast', '-crf', '26'] },
  balanced: { nvenc: ['-preset', 'p5', '-tune', 'hq', '-rc', 'vbr', '-cq', '22'], cpu: ['-preset', 'medium', '-crf', '22'] },
  quality: {
    nvenc: ['-preset', 'p7', '-tune', 'hq', '-rc', 'vbr', '-cq', '19', '-multipass', 'fullres', '-spatial-aq', '1', '-temporal-aq', '1'],
    cpu: ['-preset', 'slow', '-crf', '19'],
  },
}

export interface MontageSceneInfo {
  path: string
  durationSec: number | null
  width: number | null
  height: number | null
  fps: number | null
  hasAudio: boolean
}

export interface MontageContext {
  /** Аппаратный Vulkan (NVIDIA ICD): только тогда fruc/libplacebo идут на GPU. */
  vulkanHardware: boolean
  /** Vulkan вообще доступен (может быть программный llvmpipe). */
  vulkanAvailable: boolean
  fontFile: string
  tempDir: string
}

export interface MontagePlan extends FfmpegPlan {
  /** Вспомогательные textfile (титры счёта); сам граф идёт аргументом -filter_complex. */
  tempFiles: Array<{ path: string; content: string }>
  cleanupDir: string | null
  mode: { upscaler: 'ewa' | 'lanczos' | 'bicubic' | 'none'; interpolator: 'fruc' | 'minterpolate' | 'duplicate' | 'off' }
}

export interface FfmpegPlan {
  binary: string
  argv: string[]
  totalMs: number | null
  warnings: string[]
  usedEncoder: string | null
  usedDecoder: string | null
  operations: string[]
  concatList: { path: string; content: string } | null
}

function seconds(value: number | undefined): string {
  return String(Math.max(0, value ?? 0))
}


/** Аргументы выбранного кодировщика: NVENC на GPU, иначе CPU-фолбэк с предупреждением. */
function withCq(args: string[], cq: number | undefined): string[] {
  if (cq == null) return args
  const out = [...args]
  for (const key of ['-cq', '-crf']) {
    const index = out.indexOf(key)
    if (index >= 0) out[index + 1] = String(Math.round(cq))
  }
  return out
}

export function chooseEncoder(
  params: Pick<FfmpegParams, 'codec' | 'quality' | 'cq'>,
  caps: FfmpegCaps,
  isGpu: boolean,
  deviceIndex: number | null,
  warnings: string[],
): { encoder: string | null; args: string[]; used: string | null } {
  const codec = params.codec ?? 'hevc'
  if (codec === 'copy') return { encoder: null, args: ['-c', 'copy'], used: 'copy' }
  const gpuEncoder = NVENC_BY_CODEC[codec]
  if (isGpu && gpuEncoder && caps.encoders.includes(gpuEncoder)) {
    const quality = withCq(QUALITY[params.quality ?? 'balanced'].nvenc, params.cq)
    const extra = codec === 'hevc' ? ['-tag:v', 'hvc1'] : []
    const gpuSelector = deviceIndex != null ? ['-gpu', String(deviceIndex)] : []
    return { encoder: gpuEncoder, args: ['-c:v', gpuEncoder, ...gpuSelector, ...quality, ...extra], used: gpuEncoder }
  }
  const cpuEncoder = CPU_BY_CODEC[codec]
  if (!caps.encoders.includes(cpuEncoder)) {
    throw Object.assign(new Error(`No encoder available for codec ${codec}`), { code: 'encoder-missing' })
  }
  if (isGpu) warnings.push(`GPU encoder ${gpuEncoder} is unavailable; falling back to ${cpuEncoder}.`)
  const quality = withCq(QUALITY[params.quality ?? 'balanced'].cpu, params.cq)
  const extra = codec === 'hevc' ? ['-tag:v', 'hvc1'] : []
  return { encoder: cpuEncoder, args: ['-c:v', cpuEncoder, ...quality, ...extra], used: cpuEncoder }
}

/** Чётный размер кадра: h264/hevc NVENC не любят нечётные стороны. */
function even(value: number): number {
  const rounded = Math.max(2, Math.round(value))
  return rounded % 2 === 0 ? rounded : rounded + 1
}

/** Экранирование значения для filter_complex: одинарные кавычки и обратные слэши. */
function filterQuote(value: string): string {
  return `'${value.replace(/\\/g, '\\\\').replace(/'/g, "'\\''")}'`
}

/** Собирает полный вызов ffmpeg для одной операции. */
export function buildFfmpegPlan(
  params: FfmpegParams,
  caps: FfmpegCaps,
  settings: Settings,
  probe: ProbeInfo | null,
  deviceIndex: number | null,
): FfmpegPlan {
  const warnings: string[] = []
  if (params.cq != null && (!Number.isFinite(params.cq) || params.cq < 0 || params.cq > 51)) {
    throw Object.assign(new Error('cq must be between 0 and 51'), { code: 'invalid-args' })
  }
  const op = params.operation
  const overwrite = params.overwrite === true
  const base = ['-hide_banner', '-nostdin', '-progress', 'pipe:1', '-nostats', overwrite ? '-y' : '-n']
  const isGpu = deviceIndex !== null
  const audioArgs = (): string[] => {
    if (params.audio === 'none') return ['-an']
    if (params.audio === 'aac') return ['-c:a', 'aac', '-b:a', '192k']
    return ['-c:a', 'copy']
  }

  const pickEncoder = (): { encoder: string | null; args: string[]; used: string | null } =>
    chooseEncoder(params, caps, isGpu, deviceIndex, warnings)

  const inputArgs = (input: string): string[] => {
    const args: string[] = []
    if (isGpu && params.hwaccel !== 'off') {
      args.push('-hwaccel', 'cuda', '-hwaccel_device', String(deviceIndex))
    }
    args.push('-i', input)
    return args
  }

  const vfFilter = (): string[] => {
    if (!params.scale) return []
    return ['-vf', `scale=${params.scale.replace('x', ':')}`]
  }

  let argv: string[] = []
  let totalMs: number | null = probe?.durationSec != null ? Math.round(probe.durationSec * 1000) : null
  let usedEncoder: string | null = null
  let usedDecoder: string | null = null
  let concatList: FfmpegPlan['concatList'] = null
  const operations: string[] = []

  switch (op) {
    case 'encode': {
      if (!params.input) throw Object.assign(new Error('input is required'), { code: 'invalid-args' })
      if (!params.output) throw Object.assign(new Error('output is required'), { code: 'invalid-args' })
      const picked = pickEncoder()
      usedEncoder = picked.used
      argv.push(...base, ...inputArgs(params.input), ...vfFilter())
      if (params.fps) argv.push('-r', String(params.fps))
      argv.push(...picked.args, ...audioArgs(), '-movflags', '+faststart', params.output)
      operations.push('encode')
      break
    }
    case 'remux': {
      if (!params.input || !params.output) throw Object.assign(new Error('input and output are required'), { code: 'invalid-args' })
      argv.push(...base, '-i', params.input, '-c', 'copy', '-movflags', '+faststart', params.output)
      usedEncoder = 'copy'
      operations.push('remux')
      break
    }
    case 'trim': {
      if (!params.input || !params.output) throw Object.assign(new Error('input and output are required'), { code: 'invalid-args' })
      const start = params.startSec ?? 0
      const duration =
        params.durationSec ?? (params.endSec != null ? Math.max(0, params.endSec - start) : probe?.durationSec != null ? Math.max(0, probe.durationSec - start) : undefined)
      if (params.codec === 'copy' || params.codec === undefined) {
        argv.push(...base, '-ss', seconds(start), ...inputArgs(params.input), '-t', seconds(duration), '-c', 'copy', '-avoid_negative_ts', 'make_zero', params.output)
        usedEncoder = 'copy'
      } else {
        const picked = pickEncoder()
        usedEncoder = picked.used
        argv.push(...base, '-ss', seconds(start), ...inputArgs(params.input), '-t', seconds(duration), ...vfFilter(), ...picked.args, ...audioArgs(), params.output)
      }
      totalMs = duration != null ? Math.round(duration * 1000) : totalMs
      operations.push('trim')
      break
    }
    case 'split': {
      if (!params.input) throw Object.assign(new Error('input is required'), { code: 'invalid-args' })
      const target = params.output ?? (params.outputDir ? join(params.outputDir, 'segment_%03d.mp4') : '')
      if (!target) throw Object.assign(new Error('output or outputDir is required'), { code: 'invalid-args' })
      if (!target.includes('%')) throw Object.assign(new Error('split output must contain a %d pattern'), { code: 'invalid-args' })
      if (params.segmentSec == null && params.segmentCount == null) {
        throw Object.assign(new Error('segmentSec or segmentCount is required'), { code: 'invalid-args' })
      }
      let segmentTime = params.segmentSec
      if (segmentTime == null && params.segmentCount != null && probe?.durationSec != null) {
        segmentTime = Math.max(0.5, probe.durationSec / Math.max(1, params.segmentCount))
      }
      const reencode = params.codec !== undefined && params.codec !== 'copy'
      if (reencode) {
        const picked = pickEncoder()
        usedEncoder = picked.used
        argv.push(...base, ...inputArgs(params.input), ...vfFilter(), ...picked.args, ...audioArgs(), '-f', 'segment', '-segment_time', seconds(segmentTime), '-reset_timestamps', '1', target)
      } else {
        argv.push(...base, '-i', params.input, '-c', 'copy', '-f', 'segment', '-segment_time', seconds(segmentTime), '-reset_timestamps', '1', target)
        usedEncoder = 'copy'
      }
      operations.push('split')
      break
    }
    case 'concat': {
      if (!params.inputs?.length || !params.output) throw Object.assign(new Error('inputs and output are required'), { code: 'invalid-args' })
      const listPath = join(settings.outputRoot, `.concat-${Date.now()}.txt`)
      const content = params.inputs.map((file) => `file '${file.replace(/'/g, "'\\''")}'`).join('\n')
      concatList = { path: listPath, content }
      if (params.codec && params.codec !== 'copy') {
        const picked = pickEncoder()
        usedEncoder = picked.used
        argv.push(...base, '-f', 'concat', '-safe', '0', '-i', listPath, ...picked.args, ...audioArgs(), params.output)
      } else {
        argv.push(...base, '-f', 'concat', '-safe', '0', '-i', listPath, '-c', 'copy', params.output)
        usedEncoder = 'copy'
      }
      operations.push('concat')
      break
    }
    case 'frames': {
      if (!params.input) throw Object.assign(new Error('input is required'), { code: 'invalid-args' })
      if (!params.outputDir) throw Object.assign(new Error('outputDir is required'), { code: 'invalid-args' })
      const ext = params.format === 'jpg' ? 'jpg' : 'png'
      const fps = params.everySec && params.everySec > 0 ? 1 / params.everySec : (params.fps ?? 1)
      const start = params.startSec ?? 0
      const duration =
        params.durationSec ?? (params.endSec != null ? Math.max(0, params.endSec - start) : undefined)
      if (start > 0) argv.push('-ss', seconds(start))
      argv.push('-i', params.input)
      if (duration != null && duration > 0) argv.push('-t', seconds(duration))
      argv.push('-vf', `fps=${fps}`, '-q:v', '2', join(params.outputDir, `frame_%06d.${ext}`))
      if (duration != null) totalMs = Math.round(duration * 1000)
      operations.push('frames')
      break
    }
    case 'thumb': {
      if (!params.input || !params.output) throw Object.assign(new Error('input and output are required'), { code: 'invalid-args' })
      argv.push(...base, '-ss', seconds(params.startSec ?? 0), '-i', params.input, '-frames:v', '1')
      if (params.scale) argv.push('-vf', `scale=${params.scale.replace('x', ':')}`)
      argv.push(params.output)
      operations.push('thumb')
      break
    }
    case 'decode': {
      if (!params.input || !params.output) throw Object.assign(new Error('input and output are required'), { code: 'invalid-args' })
      argv.push(...base)
      if (isGpu && params.hwaccel !== 'off') argv.push('-hwaccel', 'cuda', '-hwaccel_device', String(deviceIndex))
      argv.push('-i', params.input, '-c:v', 'rawvideo', '-pix_fmt', 'yuv420p', '-f', 'nut', params.output)
      operations.push('decode')
      break
    }
    case 'raw': {
      const raw = (params.args ?? []).filter((item): item is string => typeof item === 'string')
      const extraInputs = (params.inputs ?? []).filter((item): item is string => typeof item === 'string' && item.length > 0)
      if (!raw.length && !extraInputs.length) {
        throw Object.assign(new Error('args or inputs is required for raw'), { code: 'invalid-args' })
      }
      const rawFlags = new Set(raw)
      argv.push('-hide_banner', '-nostdin')
      if (params.progress !== false && !rawFlags.has('-progress')) argv.push('-progress', 'pipe:1', '-nostats')
      if (!rawFlags.has('-y') && !rawFlags.has('-n')) argv.push(overwrite ? '-y' : '-n')
      for (const input of extraInputs) argv.push('-i', input)
      argv.push(...raw)
      operations.push('raw')
      const seconds = params.durationSec ?? rawDurationFromArgs(raw)
      totalMs = seconds != null && seconds > 0 ? Math.round(seconds * 1000) : null
      // Для raw энкодер выбирает сам пользователь: без явного -c:v не подставляем.
      usedEncoder = rawFlags.has('-c:v') ? String(raw[raw.indexOf('-c:v') + 1] ?? null) : null
      break
    }
    default:
      throw Object.assign(new Error(`Unsupported operation ${String(op)}`), { code: 'invalid-args' })
  }

  return {
    binary: caps.path,
    argv,
    totalMs,
    warnings,
    usedEncoder,
    usedDecoder,
    operations,
    concatList,
  }
}


/** Собирает план монтажа: concat + апскейл + fps + счёт + голос в одном ffmpeg. */
export function buildMontagePlan(
  params: FfmpegParams,
  caps: FfmpegCaps,
  settings: Settings,
  scenes: MontageSceneInfo[],
  deviceIndex: number | null,
  context: MontageContext,
): MontagePlan {
  const spec = params.montage
  if (!spec || !scenes.length) throw Object.assign(new Error('montage.scenes is required'), { code: 'invalid-args' })
  if (scenes.length > 40) {
    throw Object.assign(new Error('montage supports at most 40 scenes; pre-concat them first'), { code: 'montage-too-many-scenes' })
  }
  if (!params.output) throw Object.assign(new Error('output is required'), { code: 'invalid-args' })
  const warnings: string[] = []
  const strict = spec.strict === true
  const isGpu = deviceIndex !== null
  const profile = spec.profile === 'fast' ? 'fast' : 'best'

  const baseWidth = even(spec.normalize?.width ?? scenes[0].width ?? 1280)
  const baseHeight = even(spec.normalize?.height ?? scenes[0].height ?? 720)
  const targetWidth = even(spec.upscale?.width ?? baseWidth)
  const targetHeight = even(spec.upscale?.height ?? baseHeight)
  const baseFps = Math.max(1, Math.round(Math.max(...scenes.map((scene) => scene.fps ?? 30), 24)))
  const targetFps = spec.fps ?? 0

  // --- выбор апскейлера ---
  const hasPlacebo = caps.filters.includes('libplacebo')
  let requestedUpscaler = spec.upscale?.method ?? 'auto'
  if (requestedUpscaler === 'auto') {
    if (profile === 'fast') requestedUpscaler = 'lanczos'
    else if (hasPlacebo) requestedUpscaler = 'ewa'
    else {
      warnings.push('libplacebo is missing; EWA is replaced by lanczos')
      requestedUpscaler = 'lanczos'
    }
  }
  if (requestedUpscaler === 'ewa' && !hasPlacebo) {
    if (strict) throw Object.assign(new Error('ewa upscaling requires libplacebo in the ffmpeg build'), { code: 'montage-unsupported' })
    warnings.push('libplacebo is missing; EWA is replaced by lanczos')
    requestedUpscaler = 'lanczos'
  }
  const upscaler: MontagePlan['mode']['upscaler'] = targetWidth === baseWidth && targetHeight === baseHeight ? 'none' : (requestedUpscaler as 'ewa' | 'lanczos' | 'bicubic')

  // --- выбор интерполяции ---
  const hasFruc = caps.filters.includes('fruc_vulkan')
  const hasMinterpolate = caps.filters.includes('minterpolate')
  let requestedInterp = spec.interpolate ?? 'auto'
  if (requestedInterp === 'auto') {
    if (targetFps <= 0) requestedInterp = 'off'
    else if (profile === 'fast') requestedInterp = 'duplicate'
    else if (hasFruc && context.vulkanHardware) requestedInterp = 'fruc'
    else if (hasMinterpolate) requestedInterp = 'minterpolate'
    else {
      warnings.push('minterpolate is missing; frames are duplicated with fps')
      requestedInterp = 'duplicate'
    }
  }
  if (requestedInterp === 'fruc' && !(hasFruc && context.vulkanHardware)) {
    if (strict) {
      throw Object.assign(new Error('fruc_vulkan requires an NVIDIA Vulkan ICD with VK_NV_optical_flow'), { code: 'montage-unsupported' })
    }
    warnings.push(context.vulkanAvailable
      ? 'Vulkan runs on a software device, so fruc_vulkan is replaced by minterpolate'
      : 'Vulkan is unavailable, so fruc_vulkan is replaced by minterpolate')
    requestedInterp = hasMinterpolate ? 'minterpolate' : 'duplicate'
  }
  if (requestedInterp === 'minterpolate' && !hasMinterpolate) {
    if (strict) throw Object.assign(new Error('minterpolate filter is missing'), { code: 'montage-unsupported' })
    warnings.push('minterpolate is missing; frames are duplicated with fps')
    requestedInterp = 'duplicate'
  }
  const interpolator: MontagePlan['mode']['interpolator'] = targetFps > 0 ? (requestedInterp as MontagePlan['mode']['interpolator']) : 'off'

  const durationOf = (scene: MontageSceneInfo, index: number): number => {
    const requested = spec.scenes[index]?.durationSec
    const duration = requested ?? scene.durationSec ?? 0
    if (!(duration > 0)) throw Object.assign(new Error(`scene ${index} has no usable duration`), { code: 'invalid-args' })
    return Math.max(0.05, Math.min(duration, 6 * 3600))
  }
  const totalSec = scenes.reduce((sum, scene, index) => sum + durationOf(scene, index), 0)

  const lines: string[] = []
  const concatInputs: string[] = []
  scenes.forEach((scene, index) => {
    const duration = durationOf(scene, index)
    lines.push(
      `[${index}:v]trim=duration=${duration.toFixed(3)},setpts=PTS-STARTPTS,` +
        `scale=${baseWidth}:${baseHeight}:force_original_aspect_ratio=decrease,` +
        `pad=${baseWidth}:${baseHeight}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=${baseFps}[v${index}]`,
    )
    if (scene.hasAudio && spec.audio?.sceneAudio !== 'drop') {
      lines.push(
        `[${index}:a]atrim=duration=${duration.toFixed(3)},asetpts=PTS-STARTPTS,aresample=48000,` +
          `aformat=sample_fmts=fltp:channel_layouts=stereo[a${index}]`,
      )
    } else {
      lines.push(`anullsrc=r=48000:cl=stereo,atrim=duration=${duration.toFixed(3)}[a${index}]`)
    }
    concatInputs.push(`[v${index}][a${index}]`)
  })
  lines.push(`${concatInputs.join('')}concat=n=${scenes.length}:v=1:a=1[vc][ac]`)

  let video = 'vc'
  if (upscaler === 'ewa') {
    lines.push(`[${video}]format=nv12,hwupload,libplacebo=w=${targetWidth}:h=${targetHeight}:upscaler=ewa_lanczossharp:antiringing=0.7,hwdownload,format=nv12[vu]`)
    video = 'vu'
  } else if (upscaler !== 'none') {
    const flags = upscaler === 'bicubic' ? 'bicubic' : 'lanczos'
    lines.push(`[${video}]scale=${targetWidth}:${targetHeight}:flags=${flags}[vu]`)
    video = 'vu'
  }

  if (interpolator === 'fruc') {
    lines.push(`[${video}]format=nv12,hwupload,fruc_vulkan=fps=${targetFps},hwdownload,format=nv12[vf]`)
    video = 'vf'
  } else if (interpolator === 'minterpolate') {
    lines.push(`[${video}]minterpolate=fps=${targetFps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1[vf]`)
    video = 'vf'
  } else if (interpolator === 'duplicate') {
    lines.push(`[${video}]fps=${targetFps}[vf]`)
    video = 'vf'
  }

  const tempFiles: Array<{ path: string; content: string }> = []
  const fontFile = spec.score?.fontFile ?? context.fontFile
  if (spec.score?.events?.length) {
    if (!existsSync(fontFile)) {
      throw Object.assign(new Error(`score font not found: ${fontFile}`), { code: 'montage-font-missing' })
    }
    const fontSize = Math.round(spec.score.fontSize ?? 42)
    const margin = Math.round(spec.score.margin ?? 32)
    const color = spec.score.color ?? 'white'
    const box = spec.score.box === true
    const boxColor = spec.score.boxColor ?? 'black@0.55'
    const position = spec.score.position ?? 'top-left'
    const xExpr =
      position === 'center' ? '(w-tw)/2' : position.endsWith('right') ? `w-tw-${margin}` : String(margin)
    const yExpr =
      position === 'center' ? '(h-th)/2' : position.startsWith('bottom') ? `h-th-${margin}` : String(margin)
    spec.score.events.forEach((event, index) => {
      const textPath = join(context.tempDir, `score-${index}.txt`)
      tempFiles.push({ path: textPath, content: event.text })
      const from = Math.max(0, event.atSec)
      const until = Math.max(from + 0.1, event.untilSec ?? event.atSec + 3)
      const boxArgs = box ? `:box=1:boxcolor=${boxColor}:boxborderw=10` : ''
      lines.push(
        `[${video}]drawtext=fontfile=${filterQuote(fontFile)}:textfile=${filterQuote(textPath)}:` +
          `fontsize=${fontSize}:fontcolor=${color}:x=${xExpr}:y=${yExpr}:` +
          `enable='between(t,${from.toFixed(3)},${until.toFixed(3)})'${boxArgs}[vs${index}]`,
      )
      video = `vs${index}`
    })
  }

  const overlays = spec.overlays ?? []
  for (const [index, overlay] of overlays.entries()) {
    if (!existsSync(overlay.path)) {
      throw Object.assign(new Error(`overlay not found: ${overlay.path}`), { code: 'overlay-missing' })
    }
    if (overlay.fromSec != null && overlay.toSec != null && overlay.toSec <= overlay.fromSec) {
      throw Object.assign(new Error(`overlay ${index}: toSec must be greater than fromSec`), { code: 'invalid-args' })
    }
  }
  const subtitles = spec.subtitles
  if (subtitles?.path && !existsSync(subtitles.path)) {
    throw Object.assign(new Error(`subtitles not found: ${subtitles.path}`), { code: 'montage-subtitles-missing' })
  }

  if (subtitles?.path) {
    if (subtitles.kind === 'ass' && !caps.filters.includes('ass')) {
      if (strict) throw Object.assign(new Error('ass filter is missing in this ffmpeg build'), { code: 'montage-unsupported' })
      warnings.push('ass filter is missing; ASS subtitles are rendered through subtitles')
    }
    const filterName = subtitles.kind === 'ass' && caps.filters.includes('ass') ? 'ass' : 'subtitles'
    const parts = [`filename=${filterQuote(subtitles.path)}`]
    if (subtitles.fontsDir) parts.push(`fontsdir=${filterQuote(subtitles.fontsDir)}`)
    if (subtitles.forceStyle) parts.push(`force_style=${filterQuote(subtitles.forceStyle)}`)
    lines.push(`[${video}]${filterName}=${parts.join(':')}[vsub]`)
    video = 'vsub'
  }

  overlays.forEach((overlay, index) => {
    const inputIndex = scenes.length + index
    const width = overlay.width != null ? even(overlay.width) : null
    const height = overlay.height != null ? even(overlay.height) : null
    const opacity = Math.min(1, Math.max(0, overlay.opacity ?? 1))
    const prep = [`[${inputIndex}:v]format=rgba`]
    if (width && height) prep.push(`scale=${width}:${height}`)
    if (opacity < 1) prep.push(`colorchannelmixer=aa=${opacity.toFixed(3)}`)
    lines.push(`${prep.join(',')}[ov${index}]`)
    const enable =
      overlay.fromSec != null || overlay.toSec != null
        ? `:enable='between(t,${(overlay.fromSec ?? 0).toFixed(3)},${(overlay.toSec ?? totalSec).toFixed(3)})'`
        : ''
    lines.push(
      `[${video}][ov${index}]overlay=${Math.round(overlay.x ?? 0)}:${Math.round(overlay.y ?? 0)}:shortest=0:eof_action=repeat${enable}[vo${index}]`,
    )
    video = `vo${index}`
  })

  // Финальный пиксельный формат: NVENC и overlay дружат с yuv420p.
  lines.push(`[${video}]format=yuv420p[vout]`)
  video = 'vout'

  const tracks = spec.audio?.tracks ?? []
  let audio = 'ac'
  if (tracks.length) {
    const mixInputs: string[] = ['[ac]']
    tracks.forEach((track, index) => {
      const inputIndex = scenes.length + overlays.length + index
      const parts = [`[${inputIndex}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo`]
      if (typeof track.gainDb === 'number' && track.gainDb !== 0) parts.push(`volume=${track.gainDb}dB`)
      if (track.fadeInSec && track.fadeInSec > 0) parts.push(`afade=t=in:st=0:d=${track.fadeInSec}`)
      if (track.fadeOutSec && track.fadeOutSec > 0) parts.push(`afade=t=out:st=${Math.max(0, totalSec - track.fadeOutSec)}:d=${track.fadeOutSec}`)
      parts.push(`adelay=${Math.round(Math.max(0, track.atSec ?? 0) * 1000)}:all=1`)
      lines.push(`${parts.join(',')}[m${index}]`)
      mixInputs.push(`[m${index}]`)
    })
    const normalize = spec.audio?.normalize !== false
    lines.push(
      `${mixInputs.join('')}amix=inputs=${mixInputs.length}:duration=first:dropout_transition=0:normalize=0` +
        `${normalize ? ',loudnorm=I=-16:TP=-1.5:LRA=11' : ''}[aout]`,
    )
    audio = 'aout'
  } else if (spec.audio?.sceneAudio === 'drop') {
    lines.push(`anullsrc=r=48000:cl=stereo,atrim=duration=${totalSec.toFixed(3)}[aout]`)
    audio = 'aout'
  }

  const graph = lines.join(';')

  const base = ['-hide_banner', '-nostdin', '-progress', 'pipe:1', '-nostats', params.overwrite === true ? '-y' : '-n']
  if (upscaler === 'ewa' || interpolator === 'fruc') base.push('-init_hw_device', 'vulkan=vk', '-filter_hw_device', 'vk')
  const argv: string[] = [...base]
  scenes.forEach((scene, index) => {
    const duration = durationOf(scene, index)
    const start = Math.max(0, spec.scenes[index]?.startSec ?? 0)
    if (start > 0) argv.push('-ss', start.toFixed(3))
    argv.push('-t', duration.toFixed(3), '-i', scene.path)
  })
  overlays.forEach((overlay) => {
    const isImage = /\.(png|jpe?g|webp|bmp)$/i.test(overlay.path)
    if (overlay.loop !== false && isImage) argv.push('-loop', '1')
    argv.push('-i', overlay.path)
  })
  for (const track of tracks) argv.push('-i', track.path)
  // Не все сборки ffmpeg знают -filter_complex_script, поэтому граф идёт аргументом.
  argv.push('-filter_complex', graph, '-map', `[${video}]`, '-map', `[${audio}]`)

  const picked = chooseEncoder(params, caps, isGpu, deviceIndex, warnings)
  const bitrate = Math.max(64, Math.round(spec.audio?.bitrateKbps ?? 192))
  argv.push(...picked.args, '-c:a', 'aac', '-b:a', `${bitrate}k`, '-movflags', '+faststart', '-t', totalSec.toFixed(3), params.output)

  return {
    binary: caps.path,
    argv,
    totalMs: Math.round(totalSec * 1000),
    warnings,
    usedEncoder: picked.used,
    usedDecoder: isGpu ? 'nvdec' : 'software',
    operations: ['montage'],
    concatList: null,
    tempFiles,
    cleanupDir: context.tempDir,
    mode: { upscaler, interpolator },
  }
}

export interface ProgressState {
  outTimeUs: number | null
  frame: number | null
  fps: number | null
  speed: number | null
  totalSize: number | null
  done: boolean
}

/** Применяет строку `-progress` key=value к состоянию. */
export function applyProgressLine(state: ProgressState, line: string): boolean {
  const match = /^([a-z_]+)=(.*)$/.exec(line.trim())
  if (!match) return false
  const [, key, raw] = match
  const value = raw.trim()
  if (key === 'out_time_us' || key === 'out_time_ms') {
    // Устаревший `out_time_ms` в ffmpeg тоже задаётся в микросекундах.
    const parsed = Number(value)
    if (Number.isFinite(parsed)) state.outTimeUs = parsed
  } else if (key === 'frame') {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) state.frame = parsed
  } else if (key === 'fps') {
    const parsed = Number(value)
    if (Number.isFinite(parsed) && parsed > 0) state.fps = parsed
  } else if (key === 'speed') {
    const parsed = Number(value.replace(/x$/, ''))
    if (Number.isFinite(parsed)) state.speed = parsed
  } else if (key === 'total_size') {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) state.totalSize = parsed
  } else if (key === 'progress') {
    state.done = value === 'end'
  }
  return true
}

/** Формирует окружение ffmpeg для выбранного устройства. */
export function ffmpegEnv(settings: Settings, deviceIndex: number | null): NodeJS.ProcessEnv {
  return gpuEnv(settings, deviceIndex ?? undefined)
}

/** Создаёт каталог до записи результатов задания. */
export async function ensureOutputDir(path: string): Promise<void> {
  await mkdir(path, { recursive: true })
  await writeFile(join(path, '.keep'), '', { flag: 'a' })
}

/** Родительский каталог результата, создаётся по необходимости. */
export async function ensureParent(path: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
}
