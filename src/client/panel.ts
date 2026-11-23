/**
 * Панель GPU tool: обзор GPU и доктора, живые задания с процентом и ETA,
 * история вызовов и заданий с тем, кто их запустил, аренда GPU и настройки.
 *
 * Написана без JSX, чтобы единый IIFE-бандл Vite мог брать React из таблицы
 * модулей загрузчика DSH через `require('react')` во время выполнения.
 */

export type Language = 'ru' | 'en'

type Ui = {
  React: any
  /** Перевод: язык выбирает панель (русский по умолчанию, английский — второй). */
  t: (lang: Language, key: string) => string
  css: string
  /** Текущий язык из localStorage панели. */
  language: () => Language
  /** Смена языка: запоминается для следующих открытий панели. */
  onLanguageChange: (lang: Language) => void
}

export interface GpuInfo {
  index: number
  name: string
  totalMB: number
  usedMB: number
  freeMB: number
  driver: string
  computeCap: string
}

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
}

export interface Who {
  source: string
  sessionId: string | null
  label: string | null
  client: string | null
}

export interface Job {
  id: string
  kind: string
  operation: string
  status: string
  who: Who
  device: { kind: string; index?: number; name?: string }
  engine: string | null
  argv: string[]
  params: Record<string, unknown>
  progress: JobProgress
  createdAt: number
  startedAt: number | null
  finishedAt: number | null
  durationMs: number | null
  exitCode: number | null
  outputs: Array<{ path: string; bytes: number | null }>
  error: { code: string; message: string } | null
  logTail: string[]
}

export interface Call {
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

export interface Check {
  id: string
  title: string
  level: string
  detail: string
  fix?: string
  group: string
}

export interface SettingsView {
  ffmpegPath: string
  ffmpegGpuPath: string
  ffprobePath: string
  ffprobeGpuPath: string
  pythonPath: string
  uvPath: string
  modelsDir: string
  ocrModelDir: string
  outputRoot: string
  torchIndexUrl: string
  gpuConcurrency: number
  cpuConcurrency: number
  httpPort: number
}

export interface StateView {
  version: string
  uptimeSec: number
  apiBaseUrl: string
  mcpCommand: string
  gpu: {
    available: boolean
    passthrough: boolean
    gpus: GpuInfo[]
    processes: Array<{ gpuUuid: string; pid: number; name: string; usedMB: number }>
    error: string | null
    hints: string[]
  }
  doctor: { checks: Check[]; summary: { ok: number; warn: number; fail: number; unknown: number } }
  jobs: Job[]
  queue: { running: number; queued: number }
  leases: Array<{ id: string; holder: string; who: Who; devices: number[]; expiresAt: number }>
  calls: Call[]
  settings: SettingsView
}

function fmtDuration(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return '—'
  const total = Math.max(0, Math.round(ms / 1000))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const seconds = total % 60
  if (hours) return `${hours}h ${minutes}m`
  if (minutes) return `${minutes}m ${String(seconds).padStart(2, '0')}s`
  return `${seconds}s`
}

function fmtBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return '—'
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${bytes} B`
}

function shortSession(id: string | null): string {
  if (!id) return ''
  return id.length > 10 ? `${id.slice(0, 8)}…` : id
}

/** Создаёт компонент панели с переданными React и функцией перевода. */
export function createPanel(ui: Ui): (props: any) => any {
  const { React, t } = ui
  const h = React.createElement
  const useState = React.useState as <T>(initial: T) => [T, (value: T | ((previous: T) => T)) => void]
  const useEffect = React.useEffect as (effect: () => void | (() => void), deps?: unknown[]) => void

  function Pill(props: { status: string; tr: (key: string) => string }): any {
    const key = props.status === 'succeeded' ? 'jobs.succeeded' : `jobs.${props.status}`
    return h('span', { className: `gputool-pill gputool-pill-${props.status}` }, props.tr(key))
  }

  function Progress(props: { job: Job; tr: (key: string) => string }): any {
    const progress = props.job.progress
    const percent = progress.percent
    const determinate = typeof percent === 'number' && Number.isFinite(percent)
    const label = determinate ? `${Math.floor(percent as number)}${props.tr('progress.percent')}` : props.tr('progress.indeterminate')
    const eta = progress.etaMs != null ? `${props.tr('jobs.eta')} ${fmtDuration(progress.etaMs)}` : ''
    const counts: string[] = []
    if (progress.frame != null && progress.totalFrames != null) counts.push(`${progress.frame}/${progress.totalFrames}`)
    else if (progress.processedMs != null && progress.totalMs != null) counts.push(`${fmtDuration(progress.processedMs)} / ${fmtDuration(progress.totalMs)}`)
    if (progress.fps != null) counts.push(`${progress.fps.toFixed(1)} fps`)
    if (progress.speed != null) counts.push(`${progress.speed.toFixed(2)}x`)
    return h(
      'div',
      { className: 'gputool-progress' },
      h(
        'div',
        { className: 'gputool-progress-line' },
        h(
          'div',
          { className: `gputool-bar${determinate ? '' : ' gputool-bar-indeterminate'}`, role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': determinate ? Math.floor(percent as number) : undefined },
          h('div', { className: 'gputool-bar-fill', style: determinate ? { width: `${percent}%` } : undefined }),
        ),
        h('span', { className: 'gputool-progress-percent' }, label),
      ),
      h(
        'div',
        { className: 'gputool-progress-meta' },
        h('span', null, progress.phase || ''),
        counts.length ? h('span', null, counts.join(' · ')) : null,
        eta ? h('span', null, eta) : null,
      ),
    )
  }

  function WhoCell(props: { who: Who; tr: (key: string) => string }): any {
    const who = props.who
    const badges = h('span', { className: `gputool-source gputool-source-${who.source}` }, props.tr(`source.${who.source}`))
    if (who.sessionId) {
      return h(
        'span',
        { className: 'gputool-who', title: who.sessionId },
        h('span', { className: 'gputool-session' }, shortSession(who.sessionId)),
        who.label ? h('span', { className: 'gputool-label' }, who.label) : null,
        badges,
      )
    }
    return h(
      'span',
      { className: 'gputool-who' },
      h('span', { className: 'gputool-session' }, who.label || props.tr('jobs.external')),
      who.client ? h('span', { className: 'gputool-label' }, who.client) : null,
      badges,
    )
  }

  function Copy(props: { value: string; label?: string; tr: (key: string) => string }): any {
    const [copied, setCopied] = useState(false)
    const copy = (): void => {
      try {
        void navigator.clipboard.writeText(props.value)
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      } catch {
        /* буфер обмена недоступен */
      }
    }
    return h('button', { type: 'button', className: 'gputool-btn gputool-btn-ghost', onClick: copy, title: props.value }, copied ? props.tr('common.copied') : props.label || props.tr('common.copy'))
  }

  return function Panel(_props: any): any {
    const [language, setLanguage] = useState<Language>(ui.language())
    const tr = (key: string): string => t(language, key)
    const [state, setState] = useState<StateView | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [tab, setTab] = useState<'overview' | 'jobs' | 'history' | 'leases' | 'settings'>('overview')
    const [detail, setDetail] = useState<string | null>(null)
    const [filterText, setFilterText] = useState('')
    const [filterKind, setFilterKind] = useState<'all' | 'jobs' | 'calls'>('all')
    const [settingsDraft, setSettingsDraft] = useState<SettingsView | null>(null)
    const [savedFlag, setSavedFlag] = useState(false)

    const load = async (): Promise<void> => {
      try {
        const response = await fetch('/gputool/api/state', { headers: { accept: 'application/json' } })
        const json = await response.json()
        if (!response.ok || !json?.ok) throw new Error(json?.error?.message || `HTTP ${response.status}`)
        setState(json.value as StateView)
        setError(null)
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    }

    const refreshAll = async (): Promise<void> => {
      await load()
      try {
        await fetch('/gputool/api/gpu?refresh=1', { headers: { accept: 'application/json' } })
        await fetch('/gputool/api/doctor?refresh=1', { headers: { accept: 'application/json' } })
      } catch {
        /* игнорируем */
      }
      await load()
    }

    useEffect(() => {
      void load()
      const timer = setInterval(() => {
        if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return
        void load()
      }, 1000)
      const onVisible = (): void => {
        if (document.visibilityState === 'visible') void load()
      }
      document.addEventListener('visibilitychange', onVisible)
      return () => {
        clearInterval(timer)
        document.removeEventListener('visibilitychange', onVisible)
      }
    }, [])

    useEffect(() => {
      if (!savedFlag) return undefined
      const timer = setTimeout(() => setSavedFlag(false), 1500)
      return () => clearTimeout(timer)
    }, [savedFlag])

    const cancelJob = async (id: string): Promise<void> => {
      try {
        await fetch(`/gputool/api/jobs/${id}/cancel`, { method: 'POST' })
        await load()
      } catch {
        /* игнорируем */
      }
    }

    const releaseLease = async (id: string): Promise<void> => {
      try {
        await fetch(`/gputool/api/leases/${id}/release`, { method: 'POST' })
        await load()
      } catch {
        /* игнорируем */
      }
    }

    const saveSettings = async (draft: SettingsView): Promise<void> => {
      try {
        const response = await fetch('/gputool/api/settings', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(draft),
        })
        const json = await response.json()
        if (!response.ok || !json?.ok) throw new Error(json?.error?.message || 'failed')
        setSettingsDraft(null)
        setSavedFlag(true)
        await load()
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    }

    const renderGpu = (): any => {
      const gpu = state?.gpu
      if (!gpu) return null
      return h(
        'section',
        { className: 'gputool-card' },
        h(
          'div',
          { className: 'gputool-card-head' },
          h('h3', null, tr('gpu.title')),
          h('span', { className: `gputool-tag ${gpu.available ? 'gputool-tag-ok' : 'gputool-tag-bad'}` }, gpu.available ? tr('gpu.ready') : tr('gpu.cpuOnly')),
          h('span', { className: 'gputool-muted' }, `${tr('gpu.passthrough')}: ${gpu.passthrough ? 'ok' : 'missing'}`),
        ),
        gpu.error ? h('p', { className: 'gputool-error-text' }, gpu.error) : null,
        h(
          'div',
          { className: 'gputool-gpus' },
          ...gpu.gpus.map((entry) => {
            const usedPercent = entry.totalMB > 0 ? Math.min(100, (entry.usedMB / entry.totalMB) * 100) : 0
            return h(
              'div',
              { key: entry.index, className: 'gputool-gpu' },
              h('div', { className: 'gputool-gpu-head' }, h('strong', null, `GPU ${entry.index}`), h('span', null, entry.name), h('span', { className: 'gputool-muted' }, `${tr('gpu.driver')} ${entry.driver} · ${tr('gpu.compute')} ${entry.computeCap}`)),
              h('div', { className: 'gputool-bar' }, h('div', { className: 'gputool-bar-fill gputool-bar-gpu', style: { width: `${usedPercent}%` } })),
              h('div', { className: 'gputool-muted' }, `${Math.round(entry.usedMB / 1024)} GB ${tr('gpu.used')} · ${Math.round(entry.freeMB / 1024)} GB ${tr('gpu.free')} / ${Math.round(entry.totalMB / 1024)} GB`),
            )
          }),
        ),
        h(
          'div',
          { className: 'gputool-sub' },
          h('h4', null, tr('gpu.processes')),
          gpu.processes.length
            ? h('ul', { className: 'gputool-list' }, ...gpu.processes.map((entry, index) => h('li', { key: index }, `pid ${entry.pid} · ${entry.name} · ${Math.round(entry.usedMB)} MiB`)))
            : h('p', { className: 'gputool-muted' }, tr('gpu.noProcesses')),
        ),
        gpu.hints?.length ? h('ul', { className: 'gputool-hints' }, ...gpu.hints.map((hint, index) => h('li', { key: index }, hint))) : null,
      )
    }

    const renderDoctor = (): any => {
      const checks = state?.doctor?.checks ?? []
      return h(
        'section',
        { className: 'gputool-card' },
        h('div', { className: 'gputool-card-head' }, h('h3', null, tr('doctor.title'))),
        h(
          'div',
          { className: 'gputool-checks' },
          ...checks.map((check) =>
            h(
              'div',
              { key: check.id, className: `gputool-check gputool-check-${check.level}` },
              h('span', { className: 'gputool-check-dot' }),
              h('div', { className: 'gputool-check-body' }, h('div', { className: 'gputool-check-title' }, check.title), h('div', { className: 'gputool-check-detail' }, check.detail)),
              check.fix ? h(Copy, { value: check.fix, label: tr('common.fix'), tr }) : h('span', { className: 'gputool-check-level' }, tr(`doctor.${check.level}`)),
            ),
          ),
        ),
      )
    }

    const renderApi = (): any => {
      if (!state) return null
      return h(
        'section',
        { className: 'gputool-card' },
        h('div', { className: 'gputool-card-head' }, h('h3', null, tr('api.title'))),
        h('div', { className: 'gputool-kv' }, h('span', null, 'HTTP'), h('code', null, state.apiBaseUrl), h(Copy, { value: state.apiBaseUrl, tr })),
        h('div', { className: 'gputool-kv' }, h('span', null, tr('api.mcp')), h('code', null, state.mcpCommand), h(Copy, { value: state.mcpCommand, tr })),
      )
    }

    const renderJobs = (): any => {
      const jobs = state?.jobs ?? []
      if (!jobs.length) return h('p', { className: 'gputool-empty' }, tr('jobs.empty'))
      return h(
        'div',
        { className: 'gputool-jobs' },
        ...jobs.map((job) =>
          h(
            'div',
            { key: job.id, className: `gputool-job gputool-job-${job.status}`, onClick: () => setDetail(job.id) },
            h(
              'div',
              { className: 'gputool-job-head' },
              h(Pill, { status: job.status, tr }),
              h('strong', null, `${job.kind}/${job.operation}`),
              h(WhoCell, { who: job.who, tr }),
              h('span', { className: 'gputool-muted' }, `${job.device.kind === 'gpu' ? `GPU ${job.device.index ?? 0}` : tr('device.cpu')} · ${fmtDuration(job.durationMs ?? (job.startedAt ? Date.now() - job.startedAt : null))}`),
              job.status === 'running' || job.status === 'queued'
                ? h(
                    'button',
                    {
                      type: 'button',
                      className: 'gputool-btn gputool-btn-danger',
                      onClick: (event: any) => {
                        event.stopPropagation()
                        void cancelJob(job.id)
                      },
                    },
                    tr('common.cancel'),
                  )
                : null,
            ),
            h(Progress, { job }),
          ),
        ),
      )
    }

    const renderHistory = (): any => {
      const jobs = state?.jobs ?? []
      const calls = state?.calls ?? []
      const rows: Array<{ key: string; type: 'job' | 'call'; at: number; job?: Job; call?: Call }> = []
      if (filterKind !== 'calls') for (const job of jobs) rows.push({ key: `j-${job.id}`, type: 'job', at: job.createdAt, job })
      if (filterKind !== 'jobs') for (const call of calls) rows.push({ key: `c-${call.id}`, type: 'call', at: call.startedAt, call })
      rows.sort((a, b) => b.at - a.at)
      const search = filterText.trim().toLowerCase()
      const filtered = rows.filter((row) => {
        if (!search) return true
        const source = row.job ?? row.call
        const who = source?.who
        const haystack = [who?.sessionId, who?.label, who?.client, row.job?.operation, row.job?.kind, row.call?.tool, row.call?.target, row.job?.outputs?.[0]?.path]
          .filter(Boolean)
          .join(' ')
          .toLowerCase()
        return haystack.includes(search)
      })
      return h(
        'div',
        { className: 'gputool-history' },
        h(
          'div',
          { className: 'gputool-filters' },
          h(
            'div',
            { className: 'gputool-segmented' },
            ...(['all', 'jobs', 'calls'] as const).map((value) =>
              h('button', { key: value, type: 'button', className: `gputool-seg${filterKind === value ? ' gputool-seg-active' : ''}`, onClick: () => setFilterKind(value) }, tr(`history.${value}`)),
            ),
          ),
          h('input', { className: 'gputool-input', placeholder: tr('history.search'), value: filterText, onChange: (event: any) => setFilterText(event.target.value) }),
        ),
        filtered.length
          ? h(
              'table',
              { className: 'gputool-table' },
              h('thead', null, h('tr', null, h('th', null, tr('history.when')), h('th', null, tr('jobs.who')), h('th', null, tr('jobs.operation')), h('th', null, tr('jobs.device')), h('th', null, tr('history.duration')), h('th', null, tr('history.status')))),
              h(
                'tbody',
                null,
                ...filtered.slice(0, 200).map((row) =>
                  h(
                    'tr',
                    { key: row.key, className: 'gputool-row-click', onClick: () => setDetail(row.job?.id ?? null) },
                    h('td', { className: 'gputool-nowrap' }, new Date(row.at).toLocaleString()),
                    h('td', null, row.job ? h(WhoCell, { who: row.job.who, tr }) : row.call ? h(WhoCell, { who: row.call.who, tr }) : null),
                    h('td', null, row.job ? `${row.job.kind}/${row.job.operation}` : row.call ? `${row.call.tool}${row.call.operation ? `/${row.call.operation}` : ''}` : ''),
                    h('td', null, row.job ? (row.job.device.kind === 'gpu' ? `GPU ${row.job.device.index ?? 0}` : tr('device.cpu')) : '—'),
                    h('td', null, row.job ? fmtDuration(row.job.durationMs ?? (row.job.startedAt ? Date.now() - row.job.startedAt : null)) : row.call ? fmtDuration(row.call.durationMs) : '—'),
                    h('td', null, row.job ? h(Pill, { status: row.job.status, tr }) : row.call?.ok ? h('span', { className: 'gputool-pill gputool-pill-succeeded' }, tr('doctor.ok')) : h('span', { className: 'gputool-pill gputool-pill-failed' }, tr('doctor.fail'))),
                  ),
                ),
              ),
            )
          : h('p', { className: 'gputool-empty' }, tr('history.empty')),
      )
    }

    const renderLeases = (): any => {
      const leases = state?.leases ?? []
      if (!leases.length) return h('p', { className: 'gputool-empty' }, tr('leases.empty'))
      return h(
        'table',
        { className: 'gputool-table' },
        h('thead', null, h('tr', null, h('th', null, tr('leases.holder')), h('th', null, tr('leases.devices')), h('th', null, tr('jobs.who')), h('th', null, tr('leases.expires')), h('th', null, ''))),
        h(
          'tbody',
          null,
          ...leases.map((lease) =>
            h(
              'tr',
              { key: lease.id },
              h('td', null, lease.holder),
              h('td', null, lease.devices.map((index) => `GPU ${index}`).join(', ')),
              h('td', null, h(WhoCell, { who: lease.who, tr })),
              h('td', null, new Date(lease.expiresAt).toLocaleTimeString()),
              h('td', null, h('button', { type: 'button', className: 'gputool-btn gputool-btn-danger', onClick: () => void releaseLease(lease.id) }, tr('leases.release'))),
            ),
          ),
        ),
      )
    }

    const renderSettings = (): any => {
      const settings = state?.settings
      if (!settings) return null
      const draft = settingsDraft ?? settings
      const field = (key: keyof SettingsView, label: string, numeric = false): any =>
        h(
          'label',
          { key: key, className: 'gputool-field' },
          h('span', null, label),
          h('input', {
            className: 'gputool-input',
            value: String(draft[key] ?? ''),
            inputMode: numeric ? 'numeric' : 'text',
            onChange: (event: any) => setSettingsDraft({ ...draft, [key]: numeric ? Number(event.target.value) || 0 : event.target.value }),
          }),
        )
      return h(
        'section',
        { className: 'gputool-card' },
        h('div', { className: 'gputool-card-head' }, h('h3', null, tr('settings.title')), savedFlag ? h('span', { className: 'gputool-tag gputool-tag-ok' }, tr('common.saved')) : null),
        h(
          'div',
          { className: 'gputool-fields' },
          field('ffmpegPath', tr('settings.ffmpeg')),
          field('ffmpegGpuPath', tr('settings.ffmpegGpu')),
          field('ffprobePath', tr('settings.ffprobe')),
          field('ffprobeGpuPath', tr('settings.ffprobeGpu')),
          field('pythonPath', tr('settings.python')),
          field('uvPath', tr('settings.uv')),
          field('modelsDir', tr('settings.models')),
          field('ocrModelDir', tr('settings.ocr')),
          field('outputRoot', tr('settings.outputRoot')),
          field('torchIndexUrl', tr('settings.torchIndex')),
          field('gpuConcurrency', tr('settings.gpuConcurrency'), true),
          field('cpuConcurrency', tr('settings.cpuConcurrency'), true),
          field('httpPort', tr('settings.httpPort'), true),
        ),
        h('p', { className: 'gputool-muted' }, tr('settings.note')),
        h('button', { type: 'button', className: 'gputool-btn gputool-btn-primary', onClick: () => void saveSettings(draft) }, tr('common.save')),
      )
    }

    const renderDetail = (): any => {
      const job = detail ? state?.jobs.find((entry) => entry.id === detail) ?? null : null
      if (!job) return null
      return h(
        'div',
        { className: 'gputool-overlay', onClick: () => setDetail(null) },
        h(
          'div',
          { className: 'gputool-modal', role: 'dialog', 'aria-modal': 'true', 'aria-label': tr('jobs.detail'), onClick: (event: any) => event.stopPropagation() },
          h(
            'div',
            { className: 'gputool-modal-head' },
            h('h3', null, `${job.kind}/${job.operation}`),
            h(Pill, { status: job.status, tr }),
            h('button', { type: 'button', className: 'gputool-btn gputool-btn-ghost', onClick: () => setDetail(null) }, '✕'),
          ),
          h('div', { className: 'gputool-kv' }, h('span', null, tr('jobs.who')), h('span', null, h(WhoCell, { who: job.who, tr }))),
          h('div', { className: 'gputool-kv' }, h('span', null, tr('jobs.device')), h('span', null, job.device.kind === 'gpu' ? `GPU ${job.device.index ?? 0}${job.device.name ? ` (${job.device.name})` : ''}` : tr('device.cpu'))),
          h('div', { className: 'gputool-kv' }, h('span', null, tr('history.duration')), h('span', null, fmtDuration(job.durationMs))),
          h(Progress, { job }),
          h('h4', null, tr('jobs.argv')),
          h('pre', { className: 'gputool-pre' }, job.argv.join(' ')),
          job.outputs.length ? h('h4', null, tr('jobs.outputs')) : null,
          ...job.outputs.map((entry) => h('div', { key: entry.path, className: 'gputool-kv' }, h('code', null, entry.path), h('span', { className: 'gputool-muted' }, fmtBytes(entry.bytes)), h(Copy, { value: entry.path, tr }))),
          job.error ? h('h4', null, tr('jobs.error')) : null,
          job.error ? h('pre', { className: 'gputool-pre gputool-pre-error' }, `${job.error.code}: ${job.error.message}`) : null,
          h('h4', null, tr('jobs.log')),
          h('pre', { className: 'gputool-pre' }, (job.logTail ?? []).slice(-80).join('\n') || '—'),
        ),
      )
    }

    const tabs: Array<{ id: typeof tab; label: string }> = [
      { id: 'overview', label: tr('tab.overview') },
      { id: 'jobs', label: `${tr('tab.jobs')}${state?.queue.running ? ` (${state.queue.running})` : ''}` },
      { id: 'history', label: tr('tab.history') },
      { id: 'leases', label: tr('tab.leases') },
      { id: 'settings', label: tr('tab.settings') },
    ]

    return h(
      'div',
      { className: 'gputool-root' },
      h('style', null, ui.css),
      h(
        'header',
        { className: 'gputool-header' },
        h('div', null, h('h2', null, tr('title')), h('p', { className: 'gputool-muted' }, tr('subtitle'))),
        h(
          'div',
          { className: 'gputool-actions' },
          state ? h('span', { className: `gputool-tag ${state.gpu.available ? 'gputool-tag-ok' : 'gputool-tag-bad'}` }, state.gpu.available ? tr('gpu.ready') : tr('gpu.cpuOnly')) : null,
          h(
            'div',
            { className: 'gputool-segmented', role: 'group', 'aria-label': tr('lang.label') },
            ...(['ru', 'en'] as const).map((code) =>
              h(
                'button',
                {
                  key: code,
                  type: 'button',
                  className: `gputool-seg${language === code ? ' gputool-seg-active' : ''}`,
                  'aria-pressed': language === code,
                  title: tr(code === 'ru' ? 'lang.ru' : 'lang.en'),
                  onClick: () => {
                    if (language === code) return
                    setLanguage(code)
                    ui.onLanguageChange(code)
                  },
                },
                code === 'ru' ? 'RU' : 'EN',
              ),
            ),
          ),
          h('button', { type: 'button', className: 'gputool-btn', onClick: () => void refreshAll() }, tr('common.refresh')),
        ),
      ),
      error ? h('div', { className: 'gputool-banner' }, `${tr('status.error')}: ${error}`) : null,
      !state && !error ? h('p', { className: 'gputool-muted' }, tr('status.loading')) : null,
      state
        ? h(
            'nav',
            { className: 'gputool-tabs', role: 'tablist' },
            ...tabs.map((entry) => h('button', { key: entry.id, type: 'button', role: 'tab', 'aria-selected': tab === entry.id, className: `gputool-tab${tab === entry.id ? ' gputool-tab-active' : ''}`, onClick: () => setTab(entry.id) }, entry.label)),
          )
        : null,
      state && tab === 'overview' ? h('div', { className: 'gputool-grid' }, renderGpu(), renderDoctor(), renderApi()) : null,
      state && tab === 'jobs' ? renderJobs() : null,
      state && tab === 'history' ? renderHistory() : null,
      state && tab === 'leases' ? renderLeases() : null,
      state && tab === 'settings' ? renderSettings() : null,
      renderDetail(),
    )
  }
}
