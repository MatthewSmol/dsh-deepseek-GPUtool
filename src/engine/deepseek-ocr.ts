/**
 * OCR через DeepSeek Tools API: POST {baseUrl}/v1/tools/ocr.
 *
 * Работает по приоритету «DeepSeek → локальный PP-OCRv5»: сначала сюда, при
 * недоступности ключа, сети или API — вызывающий код переключается на локальный
 * ONNX-путь. Ключ никогда не попадает в конфиг и логи: он приходит из
 * credential-store DSH или переменной окружения.
 */
import { openAsBlob } from 'node:fs'
import { stat } from 'node:fs/promises'
import { basename } from 'node:path'

export type DeepseekPurpose = 'text_recognition' | 'rich_layout_recognition'

export interface DeepseekOcrOptions {
  /** Ключ API; пустая строка означает «ключ не настроен». */
  apiKey: string
  /** Корень API, по умолчанию https://api.deepseek.com. */
  baseUrl?: string
  /** Файл для распознавания (PDF/Word/PowerPoint/изображение). */
  file: string
  /** Назначение: извлечение текста или разметка страницы. */
  purpose?: DeepseekPurpose
  /** Таймаут одного запроса (мс). */
  timeoutMs?: number
  /** Сигнал отмены задания. */
  signal?: AbortSignal
  /** Начальный процент прогресса (0..100), чтобы шкала не прыгала. */
  startPercent?: number
}

export interface DeepseekOcrPage {
  index: number
  text: string
  finishReason: 'stop' | 'length'
}

export interface DeepseekOcrResult {
  id: string
  created: number
  purpose: string
  pageCount: number
  pages: DeepseekOcrPage[]
  usage: { inputTokens: number; outputTokens: number; totalTokens: number }
  /** Размер отправленного файла и потраченное время. */
  bytes: number
  elapsedMs: number
}

export class DeepseekOcrError extends Error {
  readonly code: string
  readonly status: number

  constructor(code: string, message: string, status = 0) {
    super(message)
    this.name = 'DeepseekOcrError'
    this.code = code
    this.status = status
  }
}

const DEFAULT_BASE = 'https://api.deepseek.com'
const DEFAULT_TIMEOUT_MS = 170_000
/** Путь служебного эндпоинта относительно корня API. */
export const OCR_PATH = '/v1/tools/ocr'
/** Поддерживаемые сервисом расширения (формат всё равно определяется по содержимому). */
export const SUPPORTED_EXTENSIONS = ['pdf', 'doc', 'docx', 'ppt', 'pptx', 'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'tif', 'tiff']
const MAX_BYTES = 100 * 1024 * 1024

function apiRoot(baseUrl?: string): string {
  return String(baseUrl || DEFAULT_BASE).trim().replace(/\/+$/, '')
}

/** Проверяет, что файл можно отправить в DeepSeek OCR. */
export async function checkOcrFile(path: string): Promise<number> {
  const info = await stat(path).catch(() => null)
  if (!info || !info.isFile()) throw new DeepseekOcrError('input-missing', `File not found: ${path}`)
  if (info.size <= 0) throw new DeepseekOcrError('input-missing', `File is empty: ${path}`)
  if (info.size > MAX_BYTES) throw new DeepseekOcrError('input-too-large', `File is larger than 100 MiB: ${path}`)
  return info.size
}

/** Разбирает ответ API, не доверяя его форме. */
export function parseOcrResponse(payload: unknown): Omit<DeepseekOcrResult, 'bytes' | 'elapsedMs'> {
  const body = (payload ?? {}) as Record<string, unknown>
  const pagesInput = Array.isArray(body.pages) ? body.pages : []
  if (!pagesInput.length) throw new DeepseekShapeError('response has no pages')
  const pages: DeepseekOcrPage[] = pagesInput.map((raw, order) => {
    const page = (raw ?? {}) as Record<string, unknown>
    return {
      index: typeof page.index === 'number' ? page.index : order,
      text: typeof page.text === 'string' ? page.text : '',
      finishReason: page.finish_reason === 'length' ? 'length' : 'stop',
    }
  })
  pages.sort((left, right) => left.index - right.index)
  const usage = (body.usage ?? {}) as Record<string, unknown>
  const number = (value: unknown): number => (typeof value === 'number' && Number.isFinite(value) ? value : 0)
  return {
    id: typeof body.id === 'string' ? body.id : '',
    created: number(body.created),
    purpose: typeof body.purpose === 'string' ? body.purpose : '',
    pageCount: typeof body.page_count === 'number' ? body.page_count : pages.length,
    pages,
    usage: {
      inputTokens: number(usage.input_tokens),
      outputTokens: number(usage.output_tokens),
      totalTokens: number(usage.total_tokens),
    },
  }
}

class DeepseekShapeError extends Error {}

/** Заголовок авторизации; ключ живёт только здесь. */
function authHeader(apiKey: string): string {
  return `Bearer ${apiKey.trim()}`
}

/** Сообщение об ошибке без секретов: токены авторизации и ключ вырезаются. */
export function redactSecret(text: string, secret: string): string {
  let out = String(text ?? '')
  const key = (secret ?? '').trim()
  if (key) out = out.split(key).join('[redacted]')
  return out
    .replace(/Bearer\s+[A-Za-z0-9._~+/=-]+/gi, 'Bearer [redacted]')
    .replace(/sk-[A-Za-z0-9_-]{8,}/g, '[redacted-key]')
}

/** Сообщение об ошибке без секретов. */
function safeText(text: string, apiKey: string): string {
  return redactSecret(text, apiKey).replace(/\s+/g, ' ').trim().slice(0, 600)
}

function failureStatus(status: number): string {
  if (status === 401 || status === 403) return 'auth-failed'
  if (status === 408) return 'timeout'
  if (status === 413) return 'input-too-large'
  if (status === 429) return 'rate-limited'
  return 'api-error'
}

/** Отправляет файл в DeepSeek OCR и возвращает текст по страницам. */
export async function runDeepseekOcr(options: DeepseekOcrOptions): Promise<DeepseekOcrResult> {
  const apiKey = (options.apiKey ?? '').trim()
  if (!apiKey) throw new DeepseekOcrError('api-key-missing', 'DeepSeek API key is not configured (DEEPSEEK_API_KEY)')
  const bytes = await checkOcrFile(options.file)
  const started = Date.now()
  const form = new FormData()
  form.set('purpose', options.purpose ?? 'text_recognition')
  // openAsBlob потоково читает файл, не загружая его целиком в память.
  form.set('file', await openAsBlob(options.file), basename(options.file))

  const timeout = AbortSignal.timeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS)
  const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout
  let response: Response
  try {
    response = await fetch(`${apiRoot(options.baseUrl)}${OCR_PATH}`, {
      method: 'POST',
      headers: { authorization: authHeader(apiKey), accept: 'application/json' },
      body: form,
      signal,
      redirect: 'error',
    })
  } catch (error) {
    if (options.signal?.aborted) throw new DeepseekCancelError()
    const message = error instanceof Error ? error.message : String(error)
    throw new DeepseekOcrError('network-error', safeText(`DeepSeek OCR request failed: ${message}`, apiKey))
  }

  const text = await response.text().catch(() => '')
  if (!response.ok) {
    throw new DeepseekOcrError(
      failureStatus(response.status),
      safeText(`DeepSeek OCR returned ${response.status}: ${text || response.statusText}`, apiKey),
      response.status,
    )
  }
  let payload: unknown
  try {
    payload = JSON.parse(text)
  } catch {
    throw new DeepseekOcrError('bad-response', safeText(`DeepSeek OCR returned non-JSON: ${text}`, apiKey), response.status)
  }
  let parsed: Omit<DeepseekOcrResult, 'bytes' | 'elapsedMs'>
  try {
    parsed = parseOcrResponse(payload)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new DeepseekOcrError('bad-response', safeText(`Unexpected DeepSeek OCR response: ${message}`, apiKey), response.status)
  }
  return { ...parsed, bytes, elapsedMs: Date.now() - started }
}

/** Отмена задания пользователем. */
class DeepseekCancelError extends DeepseekOcrError {
  constructor() {
    super('cancelled', 'DeepSeek OCR request was cancelled')
  }
}
