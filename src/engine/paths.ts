/** Раскладка файлов и атомарная запись JSON для сервиса GPU tool. */
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { chmod, mkdir, rename, unlink, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'
import { randomBytes } from 'node:crypto'

/** Домашний каталог DSH (задаёт host-процесс; иначе ~/.dsh). */
export function dshHome(): string {
  return process.env.DSH_HOME || join(homedir(), '.dsh')
}

/** Каталог данных сервиса. */
export function gputoolHome(): string {
  return process.env.GPUTOOL_HOME || join(dshHome(), 'gputool')
}

export const files = {
  home: gputoolHome,
  settings: () => join(gputoolHome(), 'settings.json'),
  endpoint: () => join(gputoolHome(), 'endpoint.json'),
  jobs: () => join(gputoolHome(), 'jobs.json'),
  history: () => join(gputoolHome(), 'history.json'),
  logs: () => join(gputoolHome(), 'logs'),
  outputs: () => join(gputoolHome(), 'outputs'),
  venv: () => join(gputoolHome(), 'venv'),
  python: () => join(gputoolHome(), 'venv', 'bin', 'python'),
}

/** Синхронное создание каталогов при старте сервиса. */
export function ensureDirsSync(): void {
  for (const dir of [gputoolHome(), files.logs(), files.outputs()]) {
    mkdirSync(dir, { recursive: true, mode: 0o755 })
  }
}

export async function ensureDirs(): Promise<void> {
  for (const dir of [gputoolHome(), files.logs(), files.outputs()]) {
    await mkdir(dir, { recursive: true, mode: 0o755 })
  }
}

/** Читает JSON; при отсутствии или ошибке возвращает `fallback`. */
export function readJsonSync<T>(path: string, fallback: T): T {
  try {
    if (!existsSync(path)) return fallback
    return JSON.parse(readFileSync(path, 'utf8')) as T
  } catch {
    return fallback
  }
}

/** Атомарная запись JSON: временный файл рядом, затем rename. */
export async function writeJsonAtomic(path: string, value: unknown, mode?: number): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  const tmp = `${path}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`
  await writeFile(tmp, `${JSON.stringify(value, null, 2)}\n`, { encoding: 'utf8', mode: mode ?? 0o644 })
  if (mode !== undefined) await chmod(tmp, mode)
  try {
    await rename(tmp, path)
  } catch (error) {
    await unlink(tmp).catch(() => undefined)
    throw error
  }
}

/** Случайный токен loopback API. */
export function newToken(): string {
  return randomBytes(24).toString('hex')
}

/** Публикуемая MCP-сервером точка, чтобы плагин DSH проксировал панель. */
export interface Endpoint {
  pid: number
  port: number
  token: string
  apiBase: string
  startedAt: number
}

export async function writeEndpoint(endpoint: Endpoint): Promise<void> {
  await writeJsonAtomic(files.endpoint(), endpoint, 0o600)
}

export function readEndpoint(): Endpoint | null {
  const value = readJsonSync<Endpoint | null>(files.endpoint(), null)
  if (!value || typeof value.port !== 'number' || typeof value.token !== 'string') return null
  return value
}
