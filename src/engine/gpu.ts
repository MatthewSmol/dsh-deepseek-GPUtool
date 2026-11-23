/** Определение и диагностика GPU (только Linux-пути, без Windows-монтирований). */
import { existsSync } from 'node:fs'
import type { GpuInfo, GpuProcess, GpuStatus } from './types.js'
import { run } from './proc.js'

const QUERY = 'index,uuid,name,memory.total,memory.used,memory.free,driver_version,compute_cap'
const PROCESS_QUERY = 'gpu_uuid,pid,process_name,used_memory'
const CANDIDATES = ['nvidia-smi', '/usr/lib/wsl/lib/nvidia-smi']

/** Разбирает строку CSV nvidia-smi для одной GPU. Имя может содержать запятые. */
export function parseGpuCsv(text: string): GpuInfo[] {
  const gpus: GpuInfo[] = []
  for (const rawLine of String(text || '').split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line) continue
    const fields = line.split(',')
    // 8 полей: index, uuid, имя (может содержать запятые), total, used, free, driver, cc
    if (fields.length < 8) continue
    const cc = (fields[fields.length - 1] ?? '').trim()
    const driver = (fields[fields.length - 2] ?? '').trim()
    const free = Number(fields[fields.length - 3])
    const used = Number(fields[fields.length - 4])
    const total = Number(fields[fields.length - 5])
    const name = fields.slice(2, fields.length - 5).join(',').trim()
    const uuid = (fields[1] ?? '').trim()
    const index = Number(fields[0])
    if (!Number.isFinite(index)) continue
    gpus.push({
      index,
      uuid: uuid && uuid !== 'N/A' ? uuid : undefined,
      name: name || 'NVIDIA GPU',
      totalMB: Number.isFinite(total) ? total : 0,
      usedMB: Number.isFinite(used) ? used : 0,
      freeMB: Number.isFinite(free) ? free : Math.max(0, (Number.isFinite(total) ? total : 0) - (Number.isFinite(used) ? used : 0)),
      driver,
      computeCap: cc,
    })
  }
  return gpus.sort((a, b) => a.index - b.index)
}

/** Разбирает CSV-вывод `--query-compute-apps`. */
export function parseProcessCsv(text: string): GpuProcess[] {
  const processes: GpuProcess[] = []
  for (const rawLine of String(text || '').split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line) continue
    const fields = line.split(',')
    if (fields.length < 4) continue
    const used = Number(fields[fields.length - 1])
    const pid = Number(fields[1])
    const name = fields.slice(2, fields.length - 1).join(',').trim()
    if (!Number.isFinite(pid)) continue
    processes.push({
      gpuUuid: (fields[0] ?? '').trim(),
      pid,
      name: name || 'process',
      usedMB: Number.isFinite(used) ? used : 0,
    })
  }
  return processes
}

async function query(command: string): Promise<{ ok: true; gpus: GpuInfo[]; processes: GpuProcess[] } | { ok: false; error: string }> {
  const [gpuResult, processResult] = await Promise.all([
    run(command, [`--query-gpu=${QUERY}`, '--format=csv,noheader,nounits'], { timeoutMs: 15_000 }),
    run(command, [`--query-compute-apps=${PROCESS_QUERY}`, '--format=csv,noheader,nounits'], { timeoutMs: 15_000 }).catch(
      () => ({ code: 1, stdout: '', stderr: '', signal: null }),
    ),
  ])
  if (gpuResult.code !== 0) {
    return { ok: false, error: (gpuResult.stderr || gpuResult.stdout || `nvidia-smi exited ${gpuResult.code}`).trim() }
  }
  const gpus = parseGpuCsv(gpuResult.stdout)
  if (!gpus.length) return { ok: false, error: 'nvidia-smi returned no GPUs' }
  return { ok: true, gpus, processes: parseProcessCsv(processResult.stdout) }
}

/** Видны ли CUDA-устройства в пространстве имён этого процесса. */
export function deviceNodesPresent(): boolean {
  return existsSync('/dev/dxg') || existsSync('/dev/nvidia0') || existsSync('/dev/nvidiactl')
}

/** Подсказки по текущей ошибке (только Linux-пути). */
function hintsFor(error: string, nodes: boolean): string[] {
  const hints: string[] = []
  const message = error.toLowerCase()
  if (message.includes('nvml') || message.includes('initialize')) {
    hints.push('Update the Windows NVIDIA driver (on Windows, not in WSL), then run `wsl --shutdown` and reopen WSL.')
    hints.push('Check that /usr/lib/wsl/lib/libcuda.so.1 exists and CUDA passthrough is enabled in .wslconfig.')
  }
  if (!nodes) {
    hints.push('/dev/dxg is missing in this process namespace, so CUDA devices are unavailable.')
    hints.push('The DSH host plugin must run where the WSL GPU devices are visible; agent bash sandbox cannot see them.')
  }
  hints.push('CPU codecs (libx264/libx265) keep ffmpeg jobs usable while the GPU is unavailable.')
  return hints
}

/** Однократно опрашивает состояние GPU. */
export async function detectGpu(): Promise<GpuStatus> {
  const checkedAt = Date.now()
  const nodes = deviceNodesPresent()
  const errors: string[] = []
  for (const command of CANDIDATES) {
    try {
      const attempt = await query(command)
      if (attempt.ok) {
        return {
          available: true,
          passthrough: nodes,
          gpus: attempt.gpus,
          processes: attempt.processes,
          error: null,
          hints: nodes ? [] : ['nvidia-smi works, but device nodes are missing for child processes.'],
          checkedAt,
          source: 'linux',
        }
      }
      errors.push(attempt.error)
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error))
    }
  }
  const message = errors[0] ?? 'nvidia-smi is unavailable'
  return {
    available: false,
    passthrough: nodes,
    gpus: [],
    processes: [],
    error: message,
    hints: hintsFor(message, nodes),
    checkedAt,
    source: 'none',
  }
}
