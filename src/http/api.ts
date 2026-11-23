/** Loopback HTTP API для прокси панели DSH и внешних инструментов. */
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import type { AddressInfo } from 'node:net'
import type { Engine, HistoryQuery } from '../engine/engine.js'
import type { Who } from '../engine/types.js'

const MAX_BODY = 4 * 1024 * 1024

interface ApiOptions {
  port: number
  token: string
  logger?: { info(message: string): void; warn(message: string): void }
}

export interface ApiHandle {
  port: number
  close: () => Promise<void>
}

class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly details?: unknown,
  ) {
    super(message)
  }
}

function readBody(req: IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const declared = Number(req.headers['content-length'] ?? 0)
    if (Number.isFinite(declared) && declared > MAX_BODY) {
      reject(new HttpError(413, 'body-too-large', 'Request body is too large'))
      return
    }
    let total = 0
    const chunks: Buffer[] = []
    req.on('data', (chunk: Buffer) => {
      total += chunk.length
      if (total > MAX_BODY) {
        req.destroy(new HttpError(413, 'body-too-large', 'Request body is too large'))
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => {
      const text = Buffer.concat(chunks).toString('utf8').trim()
      if (!text) {
        resolve({})
        return
      }
      try {
        resolve(JSON.parse(text))
      } catch {
        reject(new HttpError(400, 'invalid-json', 'Request body is not valid JSON'))
      }
    })
    req.on('error', reject)
  })
}

function send(res: ServerResponse, status: number, value: unknown): void {
  const text = JSON.stringify(value)
  res.statusCode = status
  res.setHeader('content-type', 'application/json; charset=utf-8')
  res.setHeader('cache-control', 'no-store')
  res.end(text)
}

function ok(res: ServerResponse, value: unknown): void {
  send(res, 200, { ok: true, value })
}

function fail(res: ServerResponse, error: unknown): void {
  if (error instanceof HttpError) {
    send(res, error.status, { ok: false, error: { code: error.code, message: error.message, details: error.details } })
    return
  }
  const value = error as { code?: unknown; message?: unknown; details?: unknown } | null
  send(res, 400, {
    ok: false,
    error: {
      code: typeof value?.code === 'string' ? value.code : 'bad-request',
      message: error instanceof Error ? error.message : String(error),
      details: value?.details,
    },
  })
}

function whoFrom(req: IncomingMessage, body: Record<string, unknown>): Who {
  const headerSession = req.headers['x-gputool-session']
  const headerLabel = req.headers['x-gputool-label']
  const sessionId = typeof body.session === 'string' ? body.session : typeof headerSession === 'string' ? headerSession : null
  const label = typeof body.label === 'string' ? body.label : typeof headerLabel === 'string' ? headerLabel : null
  return { source: 'http', sessionId, label, client: 'http' }
}

function isLoopback(req: IncomingMessage): boolean {
  const remote = req.socket.remoteAddress ?? ''
  return remote === '' || remote === '127.0.0.1' || remote === '::1' || remote === '::ffff:127.0.0.1'
}

/** Запускает API-сервер на loopback-адресе. */
export async function startApiServer(engine: Engine, options: ApiOptions): Promise<ApiHandle> {
  let boundPort = options.port
  const server: Server = createServer(async (req, res) => {
    try {
      if (!isLoopback(req)) throw new HttpError(403, 'forbidden', 'Loopback clients only')
      const url = new URL(req.url ?? '/', 'http://127.0.0.1')
      const path = url.pathname.replace(/\/+$/, '') || '/'
      if (path === '/health' || path === '/api/health') {
        ok(res, { startedAt: engine.startedAt, uptimeSec: Math.round((Date.now() - engine.startedAt) / 1000) })
        return
      }
      if (options.token) {
        const header = req.headers.authorization
        const bearer = typeof header === 'string' && header.startsWith('Bearer ') ? header.slice(7) : ''
        const direct = req.headers['x-gputool-token']
        if (bearer !== options.token && direct !== options.token) {
          throw new HttpError(401, 'unauthorized', 'Missing or invalid API token')
        }
      }

      const method = req.method ?? 'GET'
      const base = `http://127.0.0.1:${boundPort}`

      if (path === '/api/state' && method === 'GET') {
        const state = await engine.state()
        ok(res, { ...state, apiBaseUrl: base, mcpCommand: engine.mcpCommand() })
        return
      }
      if (path === '/api/doctor' && method === 'GET') {
        ok(res, await engine.doctorReport(url.searchParams.get('refresh') === '1'))
        return
      }
      if (path === '/api/gpu' && method === 'GET') {
        ok(res, await engine.gpu(url.searchParams.get('refresh') === '1'))
        return
      }
      if (path === '/api/jobs' && method === 'GET') {
        const limit = Number(url.searchParams.get('limit') ?? 50)
        ok(res, engine.listJobs(Number.isFinite(limit) ? limit : 50, url.searchParams.get('active') === '1'))
        return
      }
      if (path === '/api/jobs' && method === 'POST') {
        const body = (await readBody(req)) as Record<string, unknown>
        const type = String(body.type ?? '')
        const who = whoFrom(req, body)
        const meta = {
          tool: `http:${type}`,
          argsHash: engine.hash({ ...body, type: undefined }),
          label: typeof body.label === 'string' ? body.label : null,
          clientName: 'http',
          who,
        }
        const params = { ...body }
        delete params.type
        let job
        if (type === 'ffmpeg') job = await engine.submitFfmpeg(params as never, meta)
        else if (type === 'yolo') job = engine.submitYolo(params, meta)
        else if (type === 'onnx') job = engine.submitOnnx(params, meta)
        else if (type === 'face') job = engine.submitFace(params, meta)
        else if (type === 'ocr') job = engine.submitOcr(params, meta)
        else if (type === 'python') job = engine.submitPython(params, meta)
        else if (type === 'setup') job = engine.submitSetup(String(params.action ?? ''), meta)
        else throw new HttpError(400, 'unknown-type', 'type must be ffmpeg|yolo|onnx|face|ocr|python|setup')
        engine.addCall({
          tool: `http:${type}`,
          operation: typeof params.operation === 'string' ? params.operation : null,
          who,
          startedAt: Date.now(),
          durationMs: 0,
          ok: true,
          error: null,
          jobId: job.id,
          target: typeof params.output === 'string' ? params.output : typeof params.source === 'string' ? params.source : null,
        })
        ok(res, job)
        return
      }
      const jobMatch = /^\/api\/jobs\/([A-Za-z0-9-]+)(\/log|\/cancel)?$/.exec(path)
      if (jobMatch) {
        const id = jobMatch[1]
        const suffix = jobMatch[2] ?? ''
        if (suffix === '/log' && method === 'GET') {
          const job = engine.getJob(id)
          if (!job) throw new HttpError(404, 'job-not-found', 'Job not found')
          const tail = Number(url.searchParams.get('tail') ?? 200)
          ok(res, { id, logPath: job.logPath, tail: job.logTail.slice(-Math.max(1, Math.min(300, tail))) })
          return
        }
        if (suffix === '/cancel' && method === 'POST') {
          if (!engine.cancelJob(id)) throw new HttpError(409, 'not-cancellable', 'Job is not queued or running')
          ok(res, { id, cancelled: true })
          return
        }
        if (suffix === '' && method === 'GET') {
          const job = engine.getJob(id)
          if (!job) throw new HttpError(404, 'job-not-found', 'Job not found')
          ok(res, job)
          return
        }
      }
      if (path === '/api/history' && method === 'GET') {
        const query: HistoryQuery = {
          limit: Number(url.searchParams.get('limit') ?? 100) || 100,
          kind: url.searchParams.get('kind') ?? undefined,
          sessionId: url.searchParams.get('session') ?? undefined,
          operation: url.searchParams.get('operation') ?? undefined,
        }
        ok(res, engine.history(query))
        return
      }
      if (path === '/api/calls' && method === 'GET') {
        ok(res, engine.calls(Number(url.searchParams.get('limit') ?? 100) || 100))
        return
      }
      if (path === '/api/hint' && method === 'POST') {
        const body = (await readBody(req)) as Record<string, unknown>
        const hint = engine.addHint({
          tool: String(body.tool ?? ''),
          argsHash: String(body.argsHash ?? ''),
          who: {
            source: 'tool',
            sessionId: typeof body.sessionId === 'string' ? body.sessionId : null,
            label: typeof body.label === 'string' ? body.label : null,
            client: typeof body.client === 'string' ? body.client : 'dsh',
          },
        })
        ok(res, hint)
        return
      }
      if (path === '/api/leases' && method === 'GET') {
        ok(res, engine.leases())
        return
      }
      if (path === '/api/lease' && method === 'POST') {
        const body = (await readBody(req)) as Record<string, unknown>
        const gpu = await engine.gpu(false)
        const devices = Array.isArray(body.devices)
          ? (body.devices as unknown[]).map((value) => Number(value)).filter((value) => Number.isFinite(value))
          : typeof body.device === 'number'
            ? [body.device]
            : gpu.gpus.slice(0, 1).map((entry) => entry.index)
        if (!devices.length) throw new HttpError(503, 'gpu-unavailable', 'No GPU is available to lease', { error: gpu.error })
        const lease = engine.acquireLease({
          holder: String(body.holder ?? 'external'),
          ttlSec: Number(body.ttlSec ?? 600) || 600,
          devices,
          who: whoFrom(req, body),
        })
        ok(res, {
          ...lease,
          env: { CUDA_VISIBLE_DEVICES: lease.devices.join(',') },
        })
        return
      }
      const leaseMatch = /^\/api\/leases\/([A-Za-z0-9-]+)\/release$/.exec(path)
      if (leaseMatch && method === 'POST') {
        ok(res, { released: engine.releaseLease(leaseMatch[1]) })
        return
      }
      if (path === '/api/setup' && method === 'POST') {
        const body = (await readBody(req)) as Record<string, unknown>
        const action = String(body.action ?? '')
        if (!action) throw new HttpError(400, 'invalid-args', 'action is required')
        const meta = { tool: 'http:setup', argsHash: engine.hash(body), label: null, clientName: 'http', who: whoFrom(req, body) }
        if (action === 'check' || action === 'refresh-doctor') {
          ok(res, await engine.doctorReport(true))
          return
        }
        ok(res, engine.submitSetup(action, meta))
        return
      }
      if (path === '/api/settings' && method === 'GET') {
        ok(res, engine.settingsView())
        return
      }
      if (path === '/api/settings' && method === 'POST') {
        const body = (await readBody(req)) as Record<string, unknown>
        ok(res, await engine.updateSettings(body as never))
        return
      }
      throw new HttpError(404, 'not-found', `No route for ${method} ${path}`)
    } catch (error) {
      fail(res, error)
    }
  })

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(options.port, '127.0.0.1', () => resolve())
  })
  const address = server.address() as AddressInfo
  boundPort = address.port
  return {
    port: address.port,
    close: async () => {
      await new Promise<void>((resolve) => server.close(() => resolve()))
    },
  }
}
