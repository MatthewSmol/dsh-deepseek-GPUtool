#!/usr/bin/env node
/**
 * MCP-сервер GPU tool.
 *
 * Отдаёт движок GPU-задач по Model Context Protocol (stdio) и поднимает в этом
 * же процессе loopback HTTP API для прокси панели DSH и внешних инструментов.
 * GPU-процессы запускает именно этот процесс: он работает в среде хоста DSH,
 * где доступны /dev/dxg, NVENC и CUVID.
 */
import { McpServer, ResourceTemplate } from '@modelcontextprotocol/server'
import { StdioServerTransport } from '@modelcontextprotocol/server/stdio'
import { z } from 'zod'
import { Engine, VERSION } from '../engine/engine.js'
import { startApiServer } from '../http/api.js'
import { writeEndpoint, files } from '../engine/paths.js'
import type { DeviceRef, Who } from '../engine/types.js'

const logger = {
  info(message: string): void {
    process.stderr.write(`[gputool] ${message}\n`)
  },
  warn(message: string): void {
    process.stderr.write(`[gputool] warn: ${message}\n`)
  },
}

interface ToolError {
  code?: string
  message?: string
  details?: unknown
}

function asToolError(error: unknown): ToolError {
  if (error instanceof Error) {
    const value = error as Error & { code?: string; details?: unknown }
    return { code: value.code, message: value.message, details: value.details }
  }
  return { message: String(error) }
}

/** Читает CLI-аргументы вида `--flag value`. */
function argValue(flag: string): string | undefined {
  const index = process.argv.indexOf(flag)
  return index >= 0 ? process.argv[index + 1] : undefined
}

const engine = new Engine(logger)
let api: Awaited<ReturnType<typeof startApiServer>> | null = null
let apiBase = ''
let shuttingDown = false

const deviceSchema = z.union([z.literal('auto'), z.literal('cpu'), z.number().int().min(0)])

const mcp = new McpServer(
  { name: 'gputool', version: VERSION },
  {
    instructions:
      'GPU tool: shared ffmpeg (NVDEC/NVENC), ONNX Runtime CUDA, YuNet/SFace, PP-OCRv5 OCR, YOLO11 and python jobs on the local GPUs. ' +
      'Submitting a job returns its id immediately; poll `job` for live progress, percent and ETA, and `history` for previous calls. ' +
      'Pass label to describe the intended work. GPU-required tools fail with gpu-unavailable instead of silently running on CPU.',
  },
)

function text(value: unknown, isError = false): { content: Array<{ type: 'text'; text: string }>; isError?: boolean } {
  return { content: [{ type: 'text', text: JSON.stringify(value, null, 2) }], ...(isError ? { isError: true } : {}) }
}

function toolWho(name: string, args: Record<string, unknown>): Who {
  const argsHash = engine.hash(args)
  const hinted = engine.peekWho(name, argsHash)
  if (hinted) return hinted
  return {
    source: 'mcp',
    sessionId: typeof args.session === 'string' ? args.session : null,
    label: typeof args.label === 'string' ? args.label : null,
    client: 'mcp',
  }
}

function targetOf(args: Record<string, unknown>): string | null {
  for (const key of ['output', 'source', 'input', 'model', 'script', 'outputDir']) {
    const value = args[key]
    if (typeof value === 'string' && value) return value
  }
  return null
}

async function call(name: string, args: Record<string, unknown>, fn: () => Promise<unknown>): Promise<ReturnType<typeof text>> {
  const startedAt = Date.now()
  const who = toolWho(name, args)
  try {
    const value = await fn()
    const jobId = value && typeof value === 'object' && typeof (value as { id?: unknown }).id === 'string' ? (value as { id: string }).id : null
    engine.addCall({
      tool: name,
      operation: typeof args.operation === 'string' ? args.operation : null,
      who,
      startedAt,
      durationMs: Date.now() - startedAt,
      ok: true,
      error: null,
      jobId,
      target: targetOf(args),
    })
    return text(value)
  } catch (error) {
    const shape = asToolError(error)
    engine.addCall({
      tool: name,
      operation: typeof args.operation === 'string' ? args.operation : null,
      who,
      startedAt,
      durationMs: Date.now() - startedAt,
      ok: false,
      error: { code: shape.code ?? 'failed', message: shape.message ?? 'failed' },
      jobId: null,
      target: targetOf(args),
    })
    return text({ ok: false, error: shape }, true)
  }
}

const meta = (name: string, args: Record<string, unknown>) => ({
  tool: name,
  argsHash: engine.hash(args),
  label: typeof args.label === 'string' ? args.label : null,
  clientName: 'mcp',
})

mcp.registerTool(
  'status',
  {
    title: 'GPU tool status',
    description: 'GPU devices, doctor summary, queue, leases and the API endpoint of the GPU tool service.',
    inputSchema: z.object({ refresh: z.boolean().optional() }),
  },
  async (args) => {
    return await call('status', args as Record<string, unknown>, async () => {
      const [gpu, doctor] = await Promise.all([engine.gpu(args.refresh === true), engine.doctorReport(args.refresh === true)])
      return {
        version: VERSION,
        apiBaseUrl: apiBase,
        mcpCommand: engine.mcpCommand(),
        gpu,
        doctor,
        queue: engine.listJobs(50, true).length,
        leases: engine.leases(),
      }
    })
  },
)

mcp.registerTool(
  'probe',
  {
    title: 'Probe media',
    description: 'Read ffprobe metadata (duration, codecs, size, fps) for a local media file.',
    inputSchema: z.object({ path: z.string(), label: z.string().optional() }),
  },
  async (args) => await call('probe', args as Record<string, unknown>, async () => await engine.probe(args.path)),
)

mcp.registerTool(
  'ffmpeg',
  {
    title: 'Run ffmpeg',
    description:
      'Submit an ffmpeg job: encode/decode/remux/trim/split/concat/frames/thumb/raw. ' +
      'GPU encoding uses NVENC when available; hwaccel/cuvid uses NVDEC. ' +
      'Returns the job id; poll `job` for progress and percent.',
    inputSchema: z.object({
      operation: z.enum(['encode', 'decode', 'remux', 'trim', 'split', 'concat', 'frames', 'thumb', 'raw']),
      input: z.string().optional(),
      inputs: z.array(z.string()).optional(),
      output: z.string().optional(),
      outputDir: z.string().optional(),
      codec: z.enum(['h264', 'hevc', 'av1', 'copy']).optional(),
      quality: z.enum(['fast', 'balanced', 'quality']).optional(),
      scale: z.string().optional(),
      fps: z.number().optional(),
      audio: z.enum(['copy', 'aac', 'none']).optional(),
      startSec: z.number().optional(),
      endSec: z.number().optional(),
      durationSec: z.number().optional(),
      segmentSec: z.number().optional(),
      segmentCount: z.number().optional(),
      format: z.enum(['png', 'jpg']).optional(),
      everySec: z.number().optional(),
      overwrite: z.boolean().optional(),
      device: deviceSchema.optional(),
      hwaccel: z.enum(['off', 'auto', 'cuvid']).optional(),
      args: z.array(z.string()).optional(),
      label: z.string().optional(),
    }),
  },
  async (args) => await call('ffmpeg', args as Record<string, unknown>, async () => await engine.submitFfmpeg(args as never, meta('ffmpeg', args as Record<string, unknown>))),
)

mcp.registerTool(
  'yolo',
  {
    title: 'YOLO11 on GPU',
    description:
      'Run YOLO11 predict or BoT-SORT tracking on the GPU (CUDA is required unless device="cpu"). ' +
      'Use track=true with tracker="botsort.yaml" for persistent ids. Results are written as JSONL.',
    inputSchema: z.object({
      model: z.string().optional(),
      source: z.string(),
      task: z.string().optional(),
      track: z.boolean().optional(),
      tracker: z.string().optional(),
      imgsz: z.number().optional(),
      conf: z.number().optional(),
      iou: z.number().optional(),
      classes: z.array(z.number()).optional(),
      half: z.boolean().optional(),
      saveAnnotated: z.boolean().optional(),
      maxFrames: z.number().optional(),
      outputDir: z.string().optional(),
      device: deviceSchema.optional(),
      label: z.string().optional(),
    }),
  },
  async (args) => await call('yolo', args as Record<string, unknown>, async () => engine.submitYolo(args as never, meta('yolo', args as Record<string, unknown>))),
)

mcp.registerTool(
  'onnx',
  {
    title: 'ONNX Runtime CUDA',
    description:
      'Run a generic ONNX model with TensorRT/CUDA providers. Inputs are .npy arrays. ' +
      'requireGpu defaults to true: the job fails with cuda-ep-unavailable instead of silently using CPU.',
    inputSchema: z.object({
      model: z.string(),
      inputs: z.array(z.object({ name: z.string().optional(), path: z.string() })),
      providers: z.array(z.string()).optional(),
      requireGpu: z.boolean().optional(),
      outputDir: z.string().optional(),
      device: deviceSchema.optional(),
      label: z.string().optional(),
    }),
  },
  async (args) => await call('onnx', args as Record<string, unknown>, async () => engine.submitOnnx(args as never, meta('onnx', args as Record<string, unknown>))),
)

mcp.registerTool(
  'face',
  {
    title: 'YuNet + SFace face scan',
    description:
      'Scan a video with YuNet face detection and SFace embeddings on the GPU, optionally matching a reference face (cosine similarity). ' +
      'Samples at fps (default 1) and writes face.jsonl with boxes, landmarks and similarity.',
    inputSchema: z.object({
      source: z.string(),
      refImage: z.string().optional(),
      fps: z.number().optional(),
      startSec: z.number().optional(),
      durationSec: z.number().optional(),
      conf: z.number().optional(),
      matchThreshold: z.number().optional(),
      outputDir: z.string().optional(),
      device: deviceSchema.optional(),
      label: z.string().optional(),
    }),
  },
  async (args) => await call('face', args as Record<string, unknown>, async () => engine.submitFace(args as never, meta('face', args as Record<string, unknown>))),
)

mcp.registerTool(
  'ocr',
  {
    title: 'PP-OCRv5 scoreboard OCR',
    description:
      'Run PP-OCRv5 detection + recognition on the GPU for images, sampled video frames or explicit timestamps. ' +
      'region crops before detection; results are written as JSONL with text, score and boxes.',
    inputSchema: z.object({
      source: z.string().optional(),
      images: z.array(z.string()).optional(),
      atSec: z.array(z.number()).optional(),
      fps: z.number().optional(),
      startSec: z.number().optional(),
      durationSec: z.number().optional(),
      region: z.object({ x: z.number(), y: z.number(), w: z.number(), h: z.number() }).optional(),
      detLimit: z.number().optional(),
      dictPath: z.string().optional(),
      outputDir: z.string().optional(),
      device: deviceSchema.optional(),
      label: z.string().optional(),
    }),
  },
  async (args) => await call('ocr', args as Record<string, unknown>, async () => engine.submitOcr(args as never, meta('ocr', args as Record<string, unknown>))),
)

mcp.registerTool(
  'python',
  {
    title: 'Run a GPU python script',
    description:
      'Run a local python script with CUDA_VISIBLE_DEVICES and the GPU library path configured, e.g. for a tracker or RIFE stage. ' +
      'Progress lines like "progress: 42%" are mirrored into the job progress.',
    inputSchema: z.object({
      script: z.string(),
      args: z.array(z.string()).optional(),
      cwd: z.string().optional(),
      env: z.record(z.string(), z.string()).optional(),
      device: deviceSchema.optional(),
      label: z.string().optional(),
    }),
  },
  async (args) => await call('python', args as Record<string, unknown>, async () => engine.submitPython(args as never, meta('python', args as Record<string, unknown>))),
)

mcp.registerTool(
  'jobs',
  {
    title: 'List jobs',
    description: 'Recent jobs with live progress; activeOnly returns queued and running jobs.',
    inputSchema: z.object({ limit: z.number().optional(), activeOnly: z.boolean().optional() }),
  },
  async (args) => await call('jobs', args as Record<string, unknown>, async () => engine.listJobs(args.limit ?? 50, args.activeOnly === true)),
)

mcp.registerTool(
  'job',
  {
    title: 'Job details',
    description: 'One job with progress (percent, ETA, fps, frames), outputs, error and the log tail.',
    inputSchema: z.object({ jobId: z.string(), logTail: z.number().optional() }),
  },
  async (args) =>
    await call('job', args as Record<string, unknown>, async () => {
      const job = engine.getJob(args.jobId)
      if (!job) throw Object.assign(new Error('Job not found'), { code: 'job-not-found' })
      if (typeof args.logTail === 'number') job.logTail = job.logTail.slice(-Math.max(1, Math.min(300, args.logTail)))
      return job
    }),
)

mcp.registerTool(
  'cancel',
  {
    title: 'Cancel job',
    description: 'Cancel a queued or running job.',
    inputSchema: z.object({ jobId: z.string() }),
  },
  async (args) => await call('cancel', args as Record<string, unknown>, async () => ({ cancelled: engine.cancelJob(args.jobId) })),
)

mcp.registerTool(
  'history',
  {
    title: 'Call history',
    description: 'Job and tool-call history with who started it, duration, status and device.',
    inputSchema: z.object({
      limit: z.number().optional(),
      kind: z.string().optional(),
      session: z.string().optional(),
      operation: z.string().optional(),
    }),
  },
  async (args) =>
    await call('history', args as Record<string, unknown>, async () =>
      engine.history({ limit: args.limit, kind: args.kind, sessionId: args.session, operation: args.operation }),
    ),
)

mcp.registerTool(
  'gpu',
  {
    title: 'GPU devices and leases',
    description:
      'Inspect GPUs/VRAM/processes, refresh the device state, or reserve a GPU for an external process. ' +
      'A lease blocks GPU jobs on those devices until released or expired.',
    inputSchema: z.object({
      action: z.enum(['status', 'refresh', 'lease', 'release']),
      device: z.number().int().min(0).optional(),
      ttlSec: z.number().optional(),
      holder: z.string().optional(),
      leaseId: z.string().optional(),
    }),
  },
  async (args) =>
    await call('gpu', args as Record<string, unknown>, async () => {
      if (args.action === 'refresh') {
        await engine.refreshGpu()
        return await engine.gpu(false)
      }
      if (args.action === 'status') return await engine.gpu(false)
      if (args.action === 'lease') {
        const gpu = await engine.gpu(false)
        const devices = args.device != null ? [args.device] : gpu.gpus.slice(0, 1).map((entry) => entry.index)
        if (!devices.length) throw Object.assign(new Error('No GPU is available to lease'), { code: 'gpu-unavailable', details: { error: gpu.error } })
        const lease = engine.acquireLease({
          holder: args.holder ?? 'external',
          ttlSec: args.ttlSec ?? 600,
          devices,
          who: { source: 'mcp', sessionId: null, label: args.holder ?? null, client: 'mcp' },
        })
        return { ...lease, env: { CUDA_VISIBLE_DEVICES: lease.devices.join(',') } }
      }
      if (args.action === 'release') {
        if (!args.leaseId) throw Object.assign(new Error('leaseId is required'), { code: 'invalid-args' })
        return { released: engine.releaseLease(args.leaseId) }
      }
      throw Object.assign(new Error('unknown gpu action'), { code: 'invalid-args' })
    }),
)

mcp.registerTool(
  'setup',
  {
    title: 'GPU runtime setup',
    description:
      'Check capabilities or install the GPU runtime into the plugin venv: install-runtime (torch cu126 + onnxruntime-gpu + ultralytics), ' +
      'install-torch-cuda, install-onnx-gpu, refresh-doctor. The plugin never runs apt.',
    inputSchema: z.object({
      action: z.enum(['check', 'refresh-doctor', 'install-runtime', 'install-torch-cuda', 'install-onnx-gpu', 'refresh-ffmpeg']),
      label: z.string().optional(),
    }),
  },
  async (args) =>
    await call('setup', args as Record<string, unknown>, async () => {
      if (args.action === 'check' || args.action === 'refresh-doctor') return await engine.doctorReport(true)
      if (args.action === 'refresh-ffmpeg') {
        await engine.doctorReport(true)
        return await engine.doctorReport(true)
      }
      return engine.submitSetup(args.action, meta('setup', args as Record<string, unknown>))
    }),
)

mcp.registerTool(
  'settings',
  {
    title: 'Read or update settings',
    description: 'Read current paths and limits, or update them (ffmpeg binaries, python, models, OCR models, concurrency).',
    inputSchema: z.object({ patch: z.record(z.string(), z.unknown()).optional() }),
  },
  async (args) =>
    await call('settings', args as Record<string, unknown>, async () => {
      if (args.patch && Object.keys(args.patch).length) return await engine.updateSettings(args.patch as never)
      return engine.settingsView()
    }),
)

mcp.registerResource(
  'doctor',
  'gputool://doctor',
  { title: 'GPU doctor', description: 'Capability checks with remediation', mimeType: 'application/json' },
  async (uri) => ({ contents: [{ uri: uri.href, text: JSON.stringify(await engine.doctorReport(true), null, 2) }] }),
)

mcp.registerResource(
  'gpus',
  'gputool://gpus',
  { title: 'GPU devices', description: 'GPU memory and processes', mimeType: 'application/json' },
  async (uri) => ({ contents: [{ uri: uri.href, text: JSON.stringify(await engine.gpu(true), null, 2) }] }),
)

mcp.registerResource(
  'history',
  'gputool://history',
  { title: 'GPU tool history', description: 'Job and call history with who and duration', mimeType: 'application/json' },
  async (uri) => ({ contents: [{ uri: uri.href, text: JSON.stringify(engine.history({ limit: 200 }), null, 2) }] }),
)

mcp.registerResource(
  'jobs',
  'gputool://jobs',
  { title: 'GPU tool jobs', description: 'Recent jobs with live progress', mimeType: 'application/json' },
  async (uri) => ({ contents: [{ uri: uri.href, text: JSON.stringify(engine.listJobs(100), null, 2) }] }),
)

const jobTemplate = new ResourceTemplate('gputool://jobs/{id}', {
  list: async () => ({
    resources: engine.listJobs(50).map((job) => ({
      name: `job ${job.id}`,
      uri: `gputool://jobs/${job.id}`,
      mimeType: 'application/json',
      description: `${job.kind}/${job.operation} ${job.status}`,
    })),
  }),
})

mcp.registerResource(
  'job',
  jobTemplate,
  { title: 'GPU tool job', description: 'One job with progress and outputs', mimeType: 'application/json' },
  async (uri, variables) => {
    const id = String((variables as Record<string, unknown>).id ?? '')
    const job = engine.getJob(id)
    if (!job) {
      return { contents: [{ uri: uri.href, text: JSON.stringify({ error: 'job-not-found', id }) }] }
    }
    return { contents: [{ uri: uri.href, text: JSON.stringify(job, null, 2) }] }
  },
)

async function shutdown(reason: string): Promise<void> {
  if (shuttingDown) return
  shuttingDown = true
  logger.info(`shutting down: ${reason}`)
  try {
    await mcp.close()
  } catch {
    /* игнорируем */
  }
  try {
    await api?.close()
  } catch {
    /* игнорируем */
  }
  try {
    await engine.dispose()
  } catch {
    /* игнорируем */
  }
  process.exit(0)
}

async function main(): Promise<void> {
  const settings = engine.settingsStore.get()
  const port = Number(argValue('--port') ?? settings.httpPort) || 0
  await engine.boot()
  api = await startApiServer(engine, { port, token: settings.apiToken, logger })
  apiBase = `http://127.0.0.1:${api.port}`
  await writeEndpoint({
    pid: process.pid,
    port: api.port,
    token: settings.apiToken,
    apiBase,
    startedAt: Date.now(),
  })
  logger.info(`MCP stdio ready; HTTP API ${apiBase} (logs: ${files.logs()})`)
  const transport = new StdioServerTransport()
  await mcp.connect(transport)

  process.on('SIGINT', () => void shutdown('SIGINT'))
  process.on('SIGTERM', () => void shutdown('SIGTERM'))
  process.on('uncaughtException', (error) => logger.warn(`uncaught: ${error instanceof Error ? error.stack ?? error.message : String(error)}`))
  process.on('unhandledRejection', (reason) => logger.warn(`unhandled rejection: ${String(reason)}`))
}

void main().catch((error) => {
  logger.warn(`fatal: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}`)
  process.exit(1)
})

/** Оставляет импорт DeviceRef осмысленным для публикуемого типа. */
export type { DeviceRef }
