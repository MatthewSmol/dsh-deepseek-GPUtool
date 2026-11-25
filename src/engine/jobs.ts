/** Очередь заданий, аренда GPU, подсказки атрибуции и ограниченная история. */
import { randomUUID } from 'node:crypto'
import { appendFile, mkdir, readFile, unlink, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type {
  CallRecord,
  DeviceRef,
  HintRecord,
  JobContext,
  JobExecutor,
  JobKind,
  JobOutput,
  JobProgress,
  JobRecord,
  JobStatus,
  LeaseRecord,
  Who,
} from './types.js'
import { files } from './paths.js'

const START_PROGRESS: JobProgress = {
  percent: null,
  phase: 'queued',
  message: '',
  processedMs: null,
  totalMs: null,
  frame: null,
  totalFrames: null,
  fps: null,
  speed: null,
  bytes: null,
  segmentsDone: null,
  segmentsTotal: null,
  etaMs: null,
  updatedAt: 0,
  rate: null,
}

export interface SubmitInput {
  kind: JobKind
  operation: string
  params: Record<string, unknown>
  device: DeviceRef
  engine: string | null
  argv: string[]
  tool: string
  argsHash: string
  label: string | null
  clientName: string | null
  timeoutMs?: number
  /** Явная личность вызывающего (HTTP API); без поиска подсказки. */
  who?: Who
}

export interface JobManagerOptions {
  historyLimit: number
  jobHistoryLimit: number
  gpuConcurrency: number
  cpuConcurrency: number
  gpuAvailable: () => boolean
  /** Нет прогресса дольше этого времени → задание считается зависшим. */
  stallTimeoutMs?: number
  /** Сколько ждать «мёртвые pid» прежде чем пометить задание failed. */
  pidDeadGraceMs?: number
  /** Сколько ждать завершения executor после отмены, прежде чем финализировать принудительно. */
  cancelGraceMs?: number
  logger?: { info(message: string): void; warn(message: string): void }
}

function errorShape(error: unknown): { code: string; message: string; details?: unknown } {
  const value = error as { code?: unknown; message?: unknown; details?: unknown } | null
  return {
    code: typeof value?.code === 'string' ? value.code : 'failed',
    message: error instanceof Error ? error.message : String(error ?? 'unknown error'),
    details: value?.details,
  }
}

/** Состояние задания во время работы: контроллер отмены и файл лога. */
interface Runtime {
  controller: AbortController
  logLines: string[]
  logFile: string
  persistTimer: ReturnType<typeof setTimeout> | null
  /** Дочерние процессы задания (worker/ffmpeg), найденные через ctx.trackPid. */
  pids: Set<number>
  /** Время последнего прогресса/лога. */
  lastActivity: number
  /** Момент, когда все pid оказались мертвы. */
  deadSince: number | null
  /** Жёсткий дедлайн задания, если задан timeoutMs. */
  deadline: number | null
  /** Запись уже финализирована watchdog/отменой: executor больше не владеет слотом. */
  finalized: boolean
}

const DEFAULT_STALL_MS = 600_000
const DEFAULT_PID_DEAD_GRACE_MS = 10_000
const DEFAULT_CANCEL_GRACE_MS = 5_000

/**
 * Виды заданий, которые обслуживает ОДИН процесс Rust-воркера на плагин.
 * Параллельный старт второго такого задания падает с «worker is busy», поэтому
 * очередь держит их последовательно (статус остаётся queued, а не failed).
 */
const SERIAL_KINDS = new Set(['ball', 'face', 'upscale'])

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    return (error as NodeJS.ErrnoException)?.code === 'EPERM'
  }
}

/** Очередь с полосами по GPU, параллельностью CPU, арендой и историей. */
export class JobManager {
  private readonly jobs = new Map<string, JobRecord>()
  private readonly runtimes = new Map<string, Runtime>()
  private readonly executors = new Map<string, JobExecutor>()
  private readonly queue: string[] = []
  private readonly running = new Set<string>()
  private readonly busyGpu = new Map<number, number>()
  private readonly leases = new Map<string, LeaseRecord>()
  private readonly hints: HintRecord[] = []
  private calls: CallRecord[] = []
  private saveTimer: ReturnType<typeof setTimeout> | null = null
  private readonly logWaiters = new Map<string, Promise<void>>()

  constructor(private readonly options: JobManagerOptions) {}

  async load(): Promise<void> {
    await mkdir(files.logs(), { recursive: true })
    const raw = await readFile(files.jobs(), 'utf8').catch(() => '[]')
    let items: JobRecord[] = []
    try {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) items = parsed as JobRecord[]
    } catch {
      items = []
    }
    for (const job of items.slice(-this.options.jobHistoryLimit)) {
      if (!job || typeof job.id !== 'string') continue
      if (job.status === 'running' || job.status === 'queued') {
        job.status = 'failed'
        job.error = { code: 'interrupted', message: 'The GPU tool service restarted before this job finished' }
        job.finishedAt = job.finishedAt ?? Date.now()
        job.durationMs = job.startedAt != null ? job.finishedAt - job.startedAt : null
        job.progress.phase = 'interrupted'
      }
      this.jobs.set(job.id, job)
    }
    const historyRaw = await readFile(files.history(), 'utf8').catch(() => '[]')
    try {
      const parsed = JSON.parse(historyRaw)
      if (Array.isArray(parsed)) this.calls = (parsed as CallRecord[]).slice(-this.options.historyLimit)
    } catch {
      this.calls = []
    }
  }

  setLimits(historyLimit: number, jobHistoryLimit: number, gpuConcurrency: number, cpuConcurrency: number): void {
    this.options.historyLimit = historyLimit
    this.options.jobHistoryLimit = jobHistoryLimit
    this.options.gpuConcurrency = gpuConcurrency
    this.options.cpuConcurrency = cpuConcurrency
  }

  /** Регистрирует подсказку (из DSH pre-execute) для привязки следующего задания. */
  addHint(input: { tool: string; argsHash: string; who: Who }): HintRecord {
    const hint: HintRecord = { id: randomUUID(), tool: input.tool, argsHash: input.argsHash, who: input.who, at: Date.now() }
    this.hints.push(hint)
    if (this.hints.length > 200) this.hints.splice(0, this.hints.length - 200)
    return hint
  }

  /** Ищет подсказку без её удаления (для записей о вызовах). */
  peekHint(tool: string, argsHash: string): HintRecord | null {
    const now = Date.now()
    for (const hint of this.hints) {
      if (now - hint.at > 10_000) continue
      if (hint.tool === tool && hint.argsHash === argsHash) return hint
    }
    return null
  }

  private takeHint(tool: string, argsHash: string): HintRecord | null {
    const now = Date.now()
    for (let index = 0; index < this.hints.length; index += 1) {
      const hint = this.hints[index]
      if (now - hint.at > 10_000) continue
      if (hint.tool === tool && hint.argsHash === argsHash) {
        this.hints.splice(index, 1)
        return hint
      }
    }
    return null
  }

  whoFor(input: SubmitInput): { who: Who; hintId: string | null } {
    const hint = this.takeHint(input.tool, input.argsHash)
    if (hint) return { who: hint.who, hintId: hint.id }
    return {
      who: {
        source: input.clientName ? 'mcp' : 'system',
        sessionId: null,
        label: input.label,
        client: input.clientName,
      },
      hintId: null,
    }
  }

  submit(input: SubmitInput, executor: JobExecutor): JobRecord {
    const { who, hintId } = input.who ? { who: input.who, hintId: null } : this.whoFor(input)
    const job: JobRecord = {
      id: randomUUID(),
      kind: input.kind,
      operation: input.operation,
      status: 'queued',
      who,
      device: input.device,
      engine: input.engine,
      argv: input.argv,
      params: input.params,
      progress: { ...START_PROGRESS, updatedAt: Date.now() },
      createdAt: Date.now(),
      startedAt: null,
      finishedAt: null,
      durationMs: null,
      exitCode: null,
      signal: null,
      outputs: [],
      error: null,
      logTail: [],
      logPath: null,
      hintId,
    }
    const controller = new AbortController()
    const logFile = join(files.logs(), `${job.id}.log`)
    job.logPath = logFile
    job.pids = []
    const timeoutMs = input.timeoutMs
    this.runtimes.set(job.id, {
      controller,
      logLines: [],
      logFile,
      persistTimer: null,
      pids: new Set<number>(),
      lastActivity: Date.now(),
      deadSince: null,
      deadline: typeof timeoutMs === 'number' && timeoutMs > 0 ? Date.now() + timeoutMs : null,
      finalized: false,
    })
    this.executors.set(job.id, executor)
    this.jobs.set(job.id, job)
    this.trimJobs()
    this.queue.push(job.id)
    this.persist()
    this.pump()
    return this.clone(job)
  }

  /** Запускает готовые задания из очереди в порядке FIFO. */
  private pump(): void {
    for (const id of [...this.queue]) {
      const job = this.jobs.get(id)
      if (!job || job.status !== 'queued') {
        this.removeFromQueue(id)
        continue
      }
      if (!this.canStart(job)) continue
      this.removeFromQueue(id)
      void this.start(job)
    }
  }

  private removeFromQueue(id: string): void {
    const index = this.queue.indexOf(id)
    if (index >= 0) this.queue.splice(index, 1)
  }

  private canStart(job: JobRecord): boolean {
    if (SERIAL_KINDS.has(job.kind)) {
      for (const id of this.running) {
        if (this.jobs.get(id)?.kind === job.kind) return false
      }
    }
    if (job.device.kind === 'cpu') {
      const cpuRunning = [...this.running].filter((id) => this.jobs.get(id)?.device.kind === 'cpu').length
      return cpuRunning < this.options.cpuConcurrency
    }
    if (!this.options.gpuAvailable()) return false
    const index = job.device.index ?? 0
    const runningOnGpu = this.busyGpu.get(index) ?? 0
    if (runningOnGpu >= this.options.gpuConcurrency) return false
    const now = Date.now()
    for (const lease of this.leases.values()) {
      if (lease.expiresAt > now && lease.devices.includes(index)) return false
    }
    return true
  }

  private async start(job: JobRecord): Promise<void> {
    const runtime = this.runtimes.get(job.id)
    const executor = this.executors.get(job.id)
    if (!runtime || !executor) return
    job.status = 'running'
    job.startedAt = Date.now()
    job.progress.phase = 'starting'
    job.progress.updatedAt = Date.now()
    this.running.add(job.id)
    if (job.device.kind === 'gpu') {
      const index = job.device.index ?? 0
      this.busyGpu.set(index, (this.busyGpu.get(index) ?? 0) + 1)
    }
    this.persist()
    this.log(job, runtime, `job ${job.id} ${job.kind}/${job.operation} started on ${describeDevice(job.device)}`)
    const touch = (): void => {
      runtime.lastActivity = Date.now()
      runtime.deadSince = null
    }
    const context: JobContext = {
      job,
      signal: runtime.controller.signal,
      device: job.device,
      phase: (phase, message) => {
        job.progress.phase = phase
        if (message !== undefined) job.progress.message = message
        job.progress.updatedAt = Date.now()
        job.stalled = false
        touch()
      },
      progress: (patch) => {
        Object.assign(job.progress, patch)
        job.progress.updatedAt = Date.now()
        job.stalled = false
        touch()
        this.schedulePersist()
      },
      log: (line) => {
        touch()
        this.log(job, runtime, line)
      },
      output: (entry: JobOutput) => {
        job.outputs.push(entry)
      },
      trackPid: (pid: number) => {
        if (!Number.isFinite(pid) || pid <= 0) return
        runtime.pids.add(pid)
        if (!job.pids) job.pids = []
        if (!job.pids.includes(pid)) job.pids.push(pid)
        touch()
        this.schedulePersist()
      },
    }
    try {
      await executor(context)
      if (runtime.finalized) return
      if (runtime.controller.signal.aborted) {
        await this.release(job, runtime, 'cancelled', null, null, { code: 'cancelled', message: 'Job was cancelled' })
      } else {
        job.progress.percent = 100
        job.progress.phase = 'done'
        job.progress.updatedAt = Date.now()
        await this.release(job, runtime, 'succeeded', 0, null, null)
      }
    } catch (error) {
      if (runtime.finalized) return
      if (runtime.controller.signal.aborted) {
        await this.release(job, runtime, 'cancelled', null, null, { code: 'cancelled', message: 'Job was cancelled' })
      } else {
        await this.release(job, runtime, 'failed', job.exitCode, null, errorShape(error))
      }
    }
  }

  /** Финализирует запись ровно один раз и освобождает слот очереди. */
  private async release(
    job: JobRecord,
    runtime: Runtime,
    status: JobStatus,
    exitCode: number | null,
    signal: string | null,
    error: { code: string; message: string; details?: unknown } | null,
  ): Promise<void> {
    if (runtime.finalized) return
    runtime.finalized = true
    this.finish(job, status, exitCode, signal, error)
    job.stalled = false
    job.finishedAt = Date.now()
    job.durationMs = job.startedAt != null ? job.finishedAt - job.startedAt : null
    this.running.delete(job.id)
    if (job.device.kind === 'gpu') {
      const index = job.device.index ?? 0
      const next = (this.busyGpu.get(index) ?? 1) - 1
      if (next <= 0) this.busyGpu.delete(index)
      else this.busyGpu.set(index, next)
    }
    this.executors.delete(job.id)
    await this.flushLog(job.id)
    this.persist()
    this.pump()
  }

  private killPids(runtime: Runtime): void {
    for (const pid of runtime.pids) {
      if (!pidAlive(pid)) continue
      try {
        process.kill(pid, 'SIGTERM')
      } catch {
        /* уже мёртв */
      }
    }
    if (!runtime.pids.size) return
    const timer = setTimeout(() => {
      for (const pid of runtime.pids) {
        if (!pidAlive(pid)) continue
        try {
          process.kill(pid, 'SIGKILL')
        } catch {
          /* уже мёртв */
        }
      }
    }, 3_000)
    timer.unref?.()
  }

  /** Проверяет живость running-задания: pid, прогресс и дедлайн. */
  private async inspect(): Promise<string[]> {
    const now = Date.now()
    const failed: string[] = []
    for (const id of [...this.running]) {
      const job = this.jobs.get(id)
      const runtime = this.runtimes.get(id)
      if (!job || !runtime || runtime.finalized) continue
      if (runtime.deadline != null && now > runtime.deadline) {
        this.killPids(runtime)
        await this.release(job, runtime, 'failed', job.exitCode, null, {
          code: 'timeout',
          message: 'Job exceeded its timeout',
        })
        failed.push(id)
        continue
      }
      const hasPids = runtime.pids.size > 0
      const alivePids = hasPids ? [...runtime.pids].filter(pidAlive) : []
      if (hasPids) {
        runtime.deadSince = alivePids.length === 0 ? (runtime.deadSince ?? now) : null
        if (alivePids.length === 0) {
          if (now - (runtime.deadSince ?? now) > (this.options.pidDeadGraceMs ?? DEFAULT_PID_DEAD_GRACE_MS)) {
            await this.release(job, runtime, 'failed', job.exitCode, null, {
              code: 'process-died',
              message: 'The job process exited without reporting a result',
            })
            failed.push(id)
          }
          continue
        }
      }
      const stale = now - runtime.lastActivity > (this.options.stallTimeoutMs ?? DEFAULT_STALL_MS)
      if (!stale) continue
      if (!hasPids) {
        await this.release(job, runtime, 'failed', job.exitCode, null, {
          code: 'stalled',
          message: `No progress for ${Math.round((now - runtime.lastActivity) / 1000)} s`,
        })
        failed.push(id)
      } else {
        // Процесс жив, но молчит: показываем «зависло», не убиваем вслепую.
        job.stalled = true
      }
    }
    return failed
  }

  /** Ручной reap: немедленно прогнать watchdog и вернуть освобождённые задания. */
  async reap(): Promise<{ failed: string[]; running: number; queued: number }> {
    const failed = await this.inspect()
    this.pump()
    return { failed, running: this.running.size, queued: this.queue.length }
  }

  /** Живо ли running-задание (для /api/state). */
  isAlive(id: string): boolean {
    const job = this.jobs.get(id)
    const runtime = this.runtimes.get(id)
    if (!job || job.status !== 'running' || !runtime) return false
    if (runtime.finalized) return false
    if (runtime.pids.size) return [...runtime.pids].some(pidAlive)
    return !job.stalled
  }

  private finish(
    job: JobRecord,
    status: JobStatus,
    exitCode: number | null,
    signal: string | null,
    error: { code: string; message: string; details?: unknown } | null,
  ): void {
    job.status = status
    job.exitCode = exitCode
    job.signal = signal
    job.error = error
    if (status === 'cancelled') {
      job.progress.phase = 'cancelled'
      job.progress.message = 'Cancelled'
    }
    job.progress.updatedAt = Date.now()
  }

  private log(job: JobRecord, runtime: Runtime, line: string): void {
    const text = line.replace(/\s+$/, '')
    if (!text) return
    runtime.logLines.push(text)
    job.logTail.push(text)
    if (job.logTail.length > 300) job.logTail.splice(0, job.logTail.length - 300)
    const pending = appendFile(runtime.logFile, `${text}\n`).catch(() => undefined)
    this.logWaiters.set(job.id, pending.then(() => undefined))
    this.schedulePersist()
  }

  private async flushLog(id: string): Promise<void> {
    const waiter = this.logWaiters.get(id)
    if (waiter) await waiter.catch(() => undefined)
    this.logWaiters.delete(id)
  }

  private schedulePersist(): void {
    if (this.saveTimer) return
    this.saveTimer = setTimeout(() => {
      this.saveTimer = null
      this.persist()
    }, 500)
    this.saveTimer.unref?.()
  }

  private persist(): void {
    if (this.saveTimer) {
      clearTimeout(this.saveTimer)
      this.saveTimer = null
    }
    const jobs = [...this.jobs.values()].slice(-this.options.jobHistoryLimit)
    void writeFile(files.jobs(), `${JSON.stringify(jobs)}\n`, 'utf8').catch(() => undefined)
    void writeFile(files.history(), `${JSON.stringify(this.calls.slice(-this.options.historyLimit))}\n`, 'utf8').catch(() => undefined)
  }

  private trimJobs(): void {
    if (this.jobs.size <= this.options.jobHistoryLimit) return
    const removable = [...this.jobs.values()]
      .filter((job) => job.status !== 'running' && job.status !== 'queued')
      .sort((a, b) => a.createdAt - b.createdAt)
    while (this.jobs.size > this.options.jobHistoryLimit && removable.length) {
      const victim = removable.shift()
      if (victim) this.jobs.delete(victim.id)
    }
  }

  private clone(job: JobRecord): JobRecord {
    const copy = structuredClone(job)
    if (copy.status === 'running') {
      copy.alive = this.isAlive(copy.id)
      copy.stalled = copy.stalled === true
    }
    return copy
  }

  get(id: string): JobRecord | null {
    const job = this.jobs.get(id)
    return job ? this.clone(job) : null
  }

  list(limit = 50, activeOnly = false): JobRecord[] {
    return [...this.jobs.values()]
      .filter((job) => (activeOnly ? job.status === 'running' || job.status === 'queued' : true))
      .sort((a, b) => b.createdAt - a.createdAt)
      .slice(0, Math.max(1, Math.min(500, limit)))
      .map((job) => this.clone(job))
  }

  queueInfo(): { running: number; queued: number; stalled: number; aliveRunning: number } {
    let stalled = 0
    let aliveRunning = 0
    for (const id of this.running) {
      const job = this.jobs.get(id)
      if (!job) continue
      if (job.stalled) stalled += 1
      if (this.isAlive(id)) aliveRunning += 1
    }
    return { running: this.running.size, queued: this.queue.length, stalled, aliveRunning }
  }

  cancel(id: string): boolean {
    const job = this.jobs.get(id)
    if (!job) return false
    if (job.status === 'queued') {
      this.removeFromQueue(id)
      job.status = 'cancelled'
      job.finishedAt = Date.now()
      job.durationMs = null
      job.progress.phase = 'cancelled'
      job.error = { code: 'cancelled', message: 'Cancelled before start' }
      this.executors.delete(id)
      this.persist()
      return true
    }
    if (job.status !== 'running') return false
    const runtime = this.runtimes.get(id)
    if (!runtime || runtime.finalized) return false
    runtime.controller.abort()
    this.killPids(runtime)
    // Если executor не завершится сам — финализируем запись и освободим слот.
    const timer = setTimeout(() => {
      void this.release(job, runtime, 'cancelled', null, null, { code: 'cancelled', message: 'Job was cancelled' })
    }, this.options.cancelGraceMs ?? DEFAULT_CANCEL_GRACE_MS)
    timer.unref?.()
    return true
  }

  addCall(record: Omit<CallRecord, 'id'>): CallRecord {
    const full: CallRecord = { id: randomUUID(), ...record }
    this.calls.push(full)
    if (this.calls.length > this.options.historyLimit) this.calls.splice(0, this.calls.length - this.options.historyLimit)
    this.persist()
    return full
  }

  callsList(limit = 100): CallRecord[] {
    return this.calls.slice(-Math.max(1, Math.min(this.options.historyLimit, limit))).reverse()
  }

  addLease(input: { holder: string; who: Who; devices: number[]; ttlSec: number }): LeaseRecord {
    const now = Date.now()
    const lease: LeaseRecord = {
      id: randomUUID(),
      holder: input.holder,
      who: input.who,
      devices: input.devices,
      createdAt: now,
      expiresAt: now + Math.max(5, input.ttlSec) * 1000,
    }
    this.leases.set(lease.id, lease)
    return lease
  }

  releaseLease(id: string): boolean {
    return this.leases.delete(id)
  }

  leasesList(): LeaseRecord[] {
    const now = Date.now()
    for (const [id, lease] of this.leases) if (lease.expiresAt <= now) this.leases.delete(id)
    return [...this.leases.values()].sort((a, b) => a.createdAt - b.createdAt)
  }

  /** Периодически освобождает истёкшую аренду, гоняет watchdog и разблокирует очередь. */
  startSweeper(): () => void {
    const timer = setInterval(() => {
      const before = this.leases.size
      this.leasesList()
      if (this.leases.size !== before) this.pump()
      void this.inspect()
    }, 5_000)
    timer.unref?.()
    return () => clearInterval(timer)
  }

  /** Удаляет записи о вызовах по запросу. */
  clearCalls(): void {
    this.calls = []
    this.persist()
  }

  /** Удаляет логи заданий, выпавших из истории (по возможности). */
  async pruneLogs(keep = 300): Promise<void> {
    const jobs = [...this.jobs.values()].sort((a, b) => b.createdAt - a.createdAt).slice(0, keep)
    const keepIds = new Set(jobs.map((job) => job.id))
    const { readdir } = await import('node:fs/promises')
    const entries = await readdir(files.logs()).catch(() => [] as string[])
    for (const entry of entries) {
      if (!entry.endsWith('.log')) continue
      const id = entry.slice(0, -4)
      if (!keepIds.has(id)) await unlink(join(files.logs(), entry)).catch(() => undefined)
    }
  }
}

export function describeDevice(device: DeviceRef): string {
  if (device.kind === 'cpu') return 'CPU'
  return `GPU ${device.index ?? 0}${device.name ? ` (${device.name})` : ''}`
}

/** Записывает лог задания на диск (для вызывающих, которым нужен файл). */
export async function ensureLogFile(path: string): Promise<void> {
  await mkdir(join(path, '..'), { recursive: true }).catch(() => undefined)
  await writeFile(path, '', { flag: 'a' })
}
