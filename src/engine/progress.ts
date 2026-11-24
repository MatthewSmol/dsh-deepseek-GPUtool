/**
 * Разбор строк прогресса от python-скриптов: собственный протокол `progress: N%`
 * и ffmpeg-подобные строки (`out_time_ms=`, `time=`, `frame=`, `fps=`, `speed=`).
 */
export interface ParsedProgress {
  percent?: number
  processedMs?: number
  totalMs?: number
  frame?: number
  fps?: number
  speed?: number
  etaMs?: number
}

function clampPercent(value: number): number {
  return Math.max(0, Math.min(99.9, value))
}

/** HH:MM:SS(.mmm) → миллисекунды. */
export function timecodeToMs(value: string): number | null {
  const match = /^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$/.exec(value.trim())
  if (!match) return null
  const hours = Number(match[1])
  const minutes = Number(match[2])
  const seconds = Number(match[3])
  if (![hours, minutes, seconds].every(Number.isFinite)) return null
  return Math.round((hours * 3600 + minutes * 60 + seconds) * 1000)
}

/** Возвращает обновление прогресса из одной строки или null. */
export function parseProgressLine(line: string, totalMs: number | null = null): ParsedProgress | null {
  const text = line.trim()
  if (!text) return null
  const update: ParsedProgress = {}

  const percentMatch = /progress[^0-9]{0,12}(\d{1,3})(?:\.\d+)?\s*%/i.exec(text)
  if (percentMatch) {
    update.percent = clampPercent(Number(percentMatch[1]))
    return update
  }

  const outTimeUs = /(?:^|\s)out_time_us=(\d+)/.exec(text) ?? /(?:^|\s)out_time_ms=(\d+)/.exec(text)
  if (outTimeUs) {
    update.processedMs = Math.round(Number(outTimeUs[1]) / 1000)
  } else {
    const outTime = /(?:^|\s)(?:out_time|time)=([0-9:.]+)/.exec(text)
    if (outTime) {
      const ms = timecodeToMs(outTime[1])
      if (ms != null) update.processedMs = ms
    }
  }

  const frameMatch = /^frame=\s*(\d+)/.exec(text)
  if (frameMatch) update.frame = Number(frameMatch[1])
  const fpsMatch = /(?:^|\s)fps=\s*(\d+(?:\.\d+)?)/.exec(text)
  if (fpsMatch) update.fps = Number(fpsMatch[1])
  const speedMatch = /speed=\s*(\d+(?:\.\d+)?)x/.exec(text)
  if (speedMatch) update.speed = Number(speedMatch[1])

  if (update.processedMs == null && update.frame == null && update.fps == null && update.speed == null) {
    return null
  }

  if (totalMs != null && totalMs > 0 && update.processedMs != null) {
    update.totalMs = totalMs
    update.percent = clampPercent((update.processedMs / totalMs) * 100)
    if (update.speed && update.speed > 0) {
      const remaining = Math.max(0, totalMs - update.processedMs)
      update.etaMs = Math.round(remaining / update.speed)
    }
  }
  return update
}
