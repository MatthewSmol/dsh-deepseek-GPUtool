/** Проверка возможностей ffmpeg, сборка аргументов и разбор прогресса. */
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { mkdir, writeFile } from 'node:fs/promises'
import type { FfmpegCaps, Settings } from './types.js'
import { run } from './proc.js'
import { gpuEnv } from './settings.js'

export interface FfmpegParams {
  operation: 'encode' | 'decode' | 'remux' | 'trim' | 'split' | 'concat' | 'frames' | 'thumb' | 'raw'
  input?: string
  inputs?: string[]
  output?: string
  outputDir?: string
  codec?: 'h264' | 'hevc' | 'av1' | 'copy'
  quality?: 'fast' | 'balanced' | 'quality'
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

/** Собирает полный вызов ffmpeg для одной операции. */
export function buildFfmpegPlan(
  params: FfmpegParams,
  caps: FfmpegCaps,
  settings: Settings,
  probe: ProbeInfo | null,
  deviceIndex: number | null,
): FfmpegPlan {
  const warnings: string[] = []
  const op = params.operation
  const overwrite = params.overwrite === true
  const base = ['-hide_banner', '-nostdin', '-progress', 'pipe:1', '-nostats', overwrite ? '-y' : '-n']
  const isGpu = deviceIndex !== null
  const audioArgs = (): string[] => {
    if (params.audio === 'none') return ['-an']
    if (params.audio === 'aac') return ['-c:a', 'aac', '-b:a', '192k']
    return ['-c:a', 'copy']
  }

  const pickEncoder = (): { encoder: string | null; args: string[]; used: string | null } => {
    const codec = params.codec ?? 'hevc'
    if (codec === 'copy') return { encoder: null, args: ['-c', 'copy'], used: 'copy' }
    const gpuEncoder = NVENC_BY_CODEC[codec]
    if (isGpu && gpuEncoder && caps.encoders.includes(gpuEncoder)) {
      const quality = QUALITY[params.quality ?? 'balanced'].nvenc
      const extra = codec === 'hevc' ? ['-tag:v', 'hvc1'] : []
      // CUDA_VISIBLE_DEVICES не задаётся для ffmpeg: устройство выбираем по host-индексу.
      const gpuSelector = deviceIndex != null ? ['-gpu', String(deviceIndex)] : []
      return { encoder: gpuEncoder, args: ['-c:v', gpuEncoder, ...gpuSelector, ...quality, ...extra], used: gpuEncoder }
    }
    const cpuEncoder = CPU_BY_CODEC[codec]
    if (!caps.encoders.includes(cpuEncoder)) {
      throw Object.assign(new Error(`No encoder available for codec ${codec}`), { code: 'encoder-missing' })
    }
    if (isGpu) warnings.push(`GPU encoder ${gpuEncoder} is unavailable; falling back to ${cpuEncoder}.`)
    const quality = QUALITY[params.quality ?? 'balanced'].cpu
    const extra = codec === 'hevc' ? ['-tag:v', 'hvc1'] : []
    return { encoder: cpuEncoder, args: ['-c:v', cpuEncoder, ...quality, ...extra], used: cpuEncoder }
  }

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
      argv.push(...base, '-i', params.input, '-vf', `fps=${fps}`, '-q:v', '2', join(params.outputDir, `frame_%06d.${ext}`))
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
      if (!raw.length) throw Object.assign(new Error('args is required for raw'), { code: 'invalid-args' })
      argv = ['-hide_banner', overwrite ? '-y' : '-n', ...raw]
      operations.push('raw')
      totalMs = null
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
