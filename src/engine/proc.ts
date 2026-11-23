/** Запуск внешних процессов: отмена, ограниченные логи и построчные колбэки. */
import { spawn, type ChildProcess } from 'node:child_process'

export interface RunResult {
  code: number | null
  signal: NodeJS.Signals | null
  stdout: string
  stderr: string
}

export interface RunOptions {
  timeoutMs?: number
  cwd?: string
  env?: NodeJS.ProcessEnv
  signal?: AbortSignal
  maxOutput?: number
}

const DEFAULT_TIMEOUT = 30_000
const DEFAULT_MAX_OUTPUT = 1_048_576

/** Запускает короткую команду и собирает stdout/stderr с лимитом. */
export function run(command: string, args: string[] = [], options: RunOptions = {}): Promise<RunResult> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT
  const maxOutput = options.maxOutput ?? DEFAULT_MAX_OUTPUT
  return new Promise((resolve, reject) => {
    let child: ChildProcess
    try {
      child = spawn(command, args, {
        stdio: ['ignore', 'pipe', 'pipe'],
        cwd: options.cwd,
        env: options.env ?? process.env,
      })
    } catch (error) {
      reject(error)
      return
    }
    let stdout = ''
    let stderr = ''
    let settled = false
    const finish = (error: Error | null, code: number | null = 0, signal: NodeJS.Signals | null = null): void => {
      if (settled) return
      settled = true
      if (timer) clearTimeout(timer)
      options.signal?.removeEventListener('abort', onAbort)
      if (error) reject(error)
      else resolve({ code, signal, stdout, stderr })
    }
    const kill = (): void => {
      try {
        child.kill('SIGKILL')
      } catch {
        /* по возможности */
      }
    }
    const timer = setTimeout(() => {
      kill()
      finish(new Error(`${command} timed out after ${timeoutMs}ms`))
    }, Math.max(500, timeoutMs))
    const onAbort = (): void => {
      kill()
      finish(new Error('aborted'))
    }
    if (options.signal) {
      if (options.signal.aborted) onAbort()
      else options.signal.addEventListener('abort', onAbort, { once: true })
    }
    child.stdout?.on('data', (chunk: Buffer) => {
      if (stdout.length < maxOutput) stdout += chunk.toString('utf8')
    })
    child.stderr?.on('data', (chunk: Buffer) => {
      if (stderr.length < maxOutput) stderr += chunk.toString('utf8')
    })
    child.on('error', (error) => finish(error))
    child.on('close', (code, signal) => finish(null, code, signal))
  })
}

export interface StreamOptions {
  cwd?: string
  env?: NodeJS.ProcessEnv
  signal?: AbortSignal
  onStdoutLine?: (line: string) => void
  onStderrLine?: (line: string) => void
  /** Сколько последних строк лога сохранять в результате. */
  tailLines?: number
  /** Жёсткий лимит времени; при превышении процесс убивается. */
  timeoutMs?: number
}

export interface StreamResult extends RunResult {
  tail: string[]
}

/** Разбивает поток на строки и вызывает колбэк на каждую полную строку. */
function lineReader(onLine: (line: string) => void): { push: (chunk: Buffer) => void; flush: () => void } {
  let buffer = ''
  return {
    push(chunk: Buffer): void {
      buffer += chunk.toString('utf8')
      let index = buffer.indexOf('\n')
      while (index >= 0) {
        const line = buffer.slice(0, index).replace(/\r$/, '')
        buffer = buffer.slice(index + 1)
        if (line !== '') onLine(line)
        index = buffer.indexOf('\n')
      }
    },
    flush(): void {
      const line = buffer.replace(/\r$/, '')
      buffer = ''
      if (line !== '') onLine(line)
    },
  }
}

/** Запускает долгую команду с построчными колбэками и ограниченным хвостом лога. */
export function stream(command: string, args: string[], options: StreamOptions = {}): Promise<StreamResult> {
  const tailLines = options.tailLines ?? 200
  const tail: string[] = []
  const keep = (line: string): void => {
    tail.push(line)
    if (tail.length > tailLines) tail.splice(0, tail.length - tailLines)
  }
  return new Promise((resolve, reject) => {
    let child: ChildProcess
    try {
      child = spawn(command, args, {
        stdio: ['ignore', 'pipe', 'pipe'],
        cwd: options.cwd,
        env: options.env ?? process.env,
      })
    } catch (error) {
      reject(error)
      return
    }
    let stdout = ''
    let stderr = ''
    let settled = false
    const out = lineReader((line) => {
      keep(line)
      options.onStdoutLine?.(line)
    })
    const err = lineReader((line) => {
      keep(line)
      options.onStderrLine?.(line)
    })
    const finish = (error: Error | null, code: number | null = 0, signal: NodeJS.Signals | null = null): void => {
      if (settled) return
      settled = true
      if (timer) clearTimeout(timer)
      options.signal?.removeEventListener('abort', onAbort)
      out.flush()
      err.flush()
      if (error) reject(error)
      else resolve({ code, signal, stdout, stderr, tail })
    }
    const kill = (): void => {
      try {
        child.kill('SIGTERM')
      } catch {
        /* по возможности */
      }
      setTimeout(() => {
        try {
          child.kill('SIGKILL')
        } catch {
          /* по возможности */
        }
      }, 3_000).unref()
    }
    const abortError = new Error('aborted')
    const onAbort = (): void => {
      kill()
      // Промис завершит close с сигналом; жёсткий fallback защищает от зависания.
      setTimeout(() => finish(abortError), 5_000).unref()
    }
    const timer = options.timeoutMs
      ? setTimeout(() => {
          kill()
          finish(new Error(`${command} timed out after ${options.timeoutMs}ms`))
        }, Math.max(1_000, options.timeoutMs))
      : null
    if (options.signal) {
      if (options.signal.aborted) onAbort()
      else options.signal.addEventListener('abort', onAbort, { once: true })
    }
    child.stdout?.on('data', (chunk: Buffer) => {
      if (stdout.length < DEFAULT_MAX_OUTPUT) stdout += chunk.toString('utf8')
      out.push(chunk)
    })
    child.stderr?.on('data', (chunk: Buffer) => {
      if (stderr.length < DEFAULT_MAX_OUTPUT) stderr += chunk.toString('utf8')
      err.push(chunk)
    })
    child.on('error', (error) => finish(error))
    child.on('close', (code, signal) => finish(null, code, signal))
  })
}
