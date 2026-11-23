/**
 * Host-половина DSH-плагина GPU tool.
 *
 * Монтирует официальный MCP-клиент к локальному MCP-серверу `gputool`, поэтому
 * каждая сессия видит инструменты `mcp__gputool__*`, работающие на GPU через
 * host-процессы. Плагин также проксирует loopback HTTP API сервера под
 * `/gputool` для панели в сайдбаре и отправляет подсказку перед вызовом
 * (id сессии + стабильный хеш аргументов), чтобы задание было привязано к
 * сессии, которая его запустила.
 */
import { createHash } from 'node:crypto'
import { existsSync, readFileSync, realpathSync } from 'node:fs'
import { createRequire } from 'node:module'
import { request as httpRequest } from 'node:http'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

export const name = '@local/dsh-gputool'

/** Сервисы Host, от которых зависит плагин. */
export const inject = ['webServer', 'tools']

export interface PluginConfig {
  /** Пространство имён MCP-сервера; инструменты получают вид mcp__<serverName>__<tool>. */
  serverName?: string
  /** Таймаут одного вызова MCP-инструмента (мс). */
  toolCallTimeoutMs?: number
  /** Бюджет запуска дочернего MCP-процесса (мс). */
  startupTimeoutMs?: number
  /** Дополнительные переменные окружения MCP-сервера. */
  env?: Record<string, string>
}

interface Endpoint {
  pid: number
  port: number
  token: string
  apiBase: string
}

function packageRoot(): string {
  try {
    return join(dirname(fileURLToPath(import.meta.url)), '..')
  } catch {
    return process.cwd()
  }
}

function dshHome(): string {
  return process.env.DSH_HOME || join(process.env.HOME || '/home/main', '.dsh')
}

function endpointPath(): string {
  return join(process.env.GPUTOOL_HOME || join(dshHome(), 'gputool'), 'endpoint.json')
}

/**
 * Загружает официальный MCP-клиент DSH. Плагин установлен в профиль ссылкой,
 * поэтому импорт по имени может не найти пакеты установки DSH: ищем
 * `@deepseek-ai/dsh-mcp-client` рядом с исполняемым файлом dsh.
 */
export async function loadDshMcpClient(entry = process.argv[1] ?? ''): Promise<any> {
  try {
    return await import('@deepseek-ai/dsh-mcp-client')
  } catch {
    /* продолжим поиском рядом с установкой DSH */
  }
  const tried: string[] = []
  const candidates = new Set<string>()
  let base = ''
  try {
    base = entry ? realpathSync(entry) : ''
  } catch {
    base = entry
  }
  if (base) {
    let dir = dirname(base)
    for (let depth = 0; depth < 8; depth += 1) {
      candidates.add(join(dir, 'node_modules', '@deepseek-ai', 'dsh-mcp-client', 'lib', 'index.js'))
      candidates.add(join(dir, '..', 'node_modules', '@deepseek-ai', 'dsh-mcp-client', 'lib', 'index.js'))
      dir = dirname(dir)
    }
    try {
      const resolver = createRequire(join(dirname(base), 'noop.js'))
      candidates.add(resolver.resolve('@deepseek-ai/dsh-mcp-client'))
    } catch {
      /* обычный резолв недоступен */
    }
  }
  for (const path of candidates) {
    tried.push(path)
    if (!existsSync(path)) continue
    return await import(pathToFileURL(path).href)
  }
  throw new Error(`Не найден @deepseek-ai/dsh-mcp-client; проверенные пути: ${tried.join(', ')}`)
}

function readEndpoint(): Endpoint | null {
  try {
    const path = endpointPath()
    if (!existsSync(path)) return null
    const value = JSON.parse(readFileSync(path, 'utf8')) as Endpoint
    if (!value || typeof value.port !== 'number' || typeof value.token !== 'string') return null
    return value
  } catch {
    return null
  }
}

/** Должен совпадать со stableStringify + префиксом sha256 на сервере. */
export function argsHash(value: unknown): string {
  const seen = new WeakSet<object>()
  const walk = (input: unknown): unknown => {
    if (input === null || typeof input !== 'object') return input
    if (seen.has(input as object)) return null
    seen.add(input as object)
    if (Array.isArray(input)) return input.map(walk)
    const record = input as Record<string, unknown>
    const out: Record<string, unknown> = {}
    for (const key of Object.keys(record).sort()) out[key] = walk(record[key])
    return out
  }
  return createHash('sha256').update(JSON.stringify(walk(value ?? null))).digest('hex').slice(0, 16)
}

function isLoopback(req: IncomingMessage): boolean {
  const remote = req.socket.remoteAddress ?? ''
  return remote === '' || remote === '127.0.0.1' || remote === '::1' || remote === '::ffff:127.0.0.1'
}

function sendJson(res: ServerResponse, status: number, value: unknown): void {
  res.statusCode = status
  res.setHeader('content-type', 'application/json; charset=utf-8')
  res.setHeader('cache-control', 'no-store')
  res.end(JSON.stringify(value))
}

/** Отправляет подсказку в loopback API; ошибки не блокируют вызов инструмента. */
function postHint(endpoint: Endpoint, payload: Record<string, unknown>): Promise<void> {
  return new Promise((resolve) => {
    const body = Buffer.from(JSON.stringify(payload))
    const request = httpRequest(
      {
        host: '127.0.0.1',
        port: endpoint.port,
        path: '/api/hint',
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'content-length': body.length,
          authorization: `Bearer ${endpoint.token}`,
        },
        timeout: 1500,
      },
      (response) => {
        response.resume()
        response.on('end', () => resolve())
      },
    )
    request.on('error', () => resolve())
    request.on('timeout', () => {
      request.destroy()
      resolve()
    })
    request.end(body)
  })
}

interface ToolExecShape {
  name: string
  arguments: unknown
  agent?: { id?: string }
}

/** Проксирует запрос в loopback HTTP API MCP-сервера, сохраняя статус и тело. */
function proxy(req: IncomingMessage, res: ServerResponse): void {
  const endpoint = readEndpoint()
  if (!endpoint) {
    sendJson(res, 503, {
      ok: false,
      error: { code: 'starting', message: 'The GPU tool MCP server is starting; retry in a moment.' },
    })
    res.setHeader('retry-after', '2')
    return
  }
  const url = new URL(req.url ?? '/', 'http://127.0.0.1')
  const path = `${url.pathname.replace(/^\/gputool/, '') || '/'}${url.search}`
  const headers: Record<string, string | string[] | undefined> = { ...req.headers }
  headers.authorization = `Bearer ${endpoint.token}`
  headers.host = '127.0.0.1'
  const forwarded = httpRequest(
    {
      host: '127.0.0.1',
      port: endpoint.port,
      path,
      method: req.method,
      headers,
      timeout: 30_000,
    },
    (response) => {
      res.statusCode = response.statusCode ?? 502
      for (const [key, value] of Object.entries(response.headers)) {
        if (value !== undefined) res.setHeader(key, value)
      }
      response.pipe(res)
    },
  )
  forwarded.on('timeout', () => {
    forwarded.destroy(new Error('timeout'))
  })
  forwarded.on('error', (error) => {
    if (res.headersSent) {
      res.end()
      return
    }
    sendJson(res, 502, { ok: false, error: { code: 'proxy-failed', message: error.message } })
  })
  req.pipe(forwarded)
}

export async function apply(ctx: any, config: PluginConfig = {}): Promise<void> {
  const serverName = typeof config.serverName === 'string' && config.serverName ? config.serverName : 'gputool'
  const toolCallTimeoutMs = typeof config.toolCallTimeoutMs === 'number' ? config.toolCallTimeoutMs : 900_000
  const startupTimeoutMs = typeof config.startupTimeoutMs === 'number' ? config.startupTimeoutMs : 30_000
  const logger = typeof ctx.logger === 'function' ? ctx.logger('gputool') : { info() {}, warn: console.warn }

  const serverEntry = join(packageRoot(), 'dist', 'mcp', 'index.js')
  const mcp = await loadDshMcpClient()
  let fiber: { dispose?: () => void } | undefined
  const mount = (reason: string): void => {
    fiber?.dispose?.()
    fiber = ctx.plugin(mcp, {
      transport: 'stdio',
      serverName,
      command: process.execPath,
      args: [serverEntry, '--stdio'],
      env: {
        ...(config.env ?? {}),
        DSH_HOME: process.env.DSH_HOME ?? dshHome(),
      },
      toolCallTimeoutMs,
      startupTimeoutMs,
      failOnStartupError: false,
      reconnect: { enabled: true },
    })
    logger.info(`MCP client mounted (${reason}); prefix mcp__${serverName}__`)
  }

  mount('initial')
  ctx.effect(() => () => {
    try {
      fiber?.dispose?.()
    } catch {
      /* игнорируем */
    }
  })

  const offRoute = ctx.webServer.register({
    kind: 'prefix',
    path: '/gputool',
    handler: (req: IncomingMessage, res: ServerResponse) => {
      if (!isLoopback(req)) {
        sendJson(res, 403, { ok: false, error: { code: 'forbidden', message: 'Loopback clients only' } })
        return
      }
      proxy(req, res)
    },
  })
  ctx.effect(() => () => {
    try {
      offRoute?.()
    } catch {
      /* игнорируем */
    }
  })

  // Waterfall: привязывает предстоящий MCP-вызов к сессии до его отправки на сервер.
  ctx.on('tools/pre-execute', async (exec: ToolExecShape, next: () => Promise<unknown>) => {
    try {
      if (typeof exec?.name === 'string' && exec.name.startsWith(`mcp__${serverName}__`)) {
        const endpoint = readEndpoint()
        if (endpoint) {
          const tool = exec.name.slice(`mcp__${serverName}__`.length)
          const args = (exec.arguments && typeof exec.arguments === 'object' ? exec.arguments : {}) as Record<string, unknown>
          await postHint(endpoint, {
            tool,
            argsHash: argsHash(args),
            sessionId: exec.agent?.id ?? null,
            label: typeof args.label === 'string' ? args.label : null,
            client: 'dsh',
          })
        }
      }
    } catch (error) {
      logger.warn(`hint failed: ${error instanceof Error ? error.message : String(error)}`)
    }
    return await next()
  })

  // Изменение настроек перемонтирует клиент, чтобы MCP-процесс перезапустился с новым окружением.
  ctx.on('loader/volatile-update', () => mount('after a settings change'))

  logger.info(`GPU tool plugin ready; API proxy /gputool, server ${serverEntry}`)
}
