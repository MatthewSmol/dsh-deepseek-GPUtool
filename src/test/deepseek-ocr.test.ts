/**
 * Тесты шлюза DeepSeek OCR: разбор ответа API и проверка входа.
 *
 * Сетевые вызовы подменяются локальным сервером на 127.0.0.1, потому что ключ
 * в тестах не используется и запросы во внешний API недопустимы.
 */
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import type { AddressInfo } from 'node:net'
import { mkdtemp, rm, truncate, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { checkOcrFile, parseOcrResponse, runDeepseekOcr } from '../engine/deepseek-ocr.js'

test('parses pages, sorts them and normalizes usage', () => {
  const parsed = parseOcrResponse({
    id: 'ocr-1',
    created: 1_800_000_000,
    purpose: 'text_recognition',
    page_count: 2,
    pages: [
      { index: 1, text: 'Табло 2', finish_reason: 'length' },
      { index: 0, text: 'Табло 1', finish_reason: 'stop' },
    ],
    usage: { input_tokens: 10, output_tokens: 20, total_tokens: 30 },
  })
  assert.equal(parsed.pageCount, 2)
  assert.deepEqual(parsed.pages.map((page) => page.index), [0, 1])
  assert.equal(parsed.pages[1].finishReason, 'length')
  assert.equal(parsed.usage.totalTokens, 30)
})

test('rejects a response without pages', () => {
  assert.throws(() => parseOcrResponse({ id: 'ocr-x', pages: [] }), /no pages/)
  assert.throws(() => parseOcrResponse(null), /no pages/)
})

test('checks that the OCR input exists and is not too large', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'gputool-ocr-'))
  try {
    const file = join(dir, 'board.png')
    await writeFile(file, 'png')
    assert.equal(await checkOcrFile(file), 3)
    const empty = join(dir, 'empty.png')
    await writeFile(empty, '')
    await assert.rejects(checkOcrFile(empty), /empty/)
    const huge = join(dir, 'huge.pdf')
    await writeFile(huge, '')
    await truncate(huge, 101 * 1024 * 1024)
    await assert.rejects(checkOcrFile(huge), /100 MiB/)
    await assert.rejects(checkOcrFile(join(dir, 'missing.png')), /not found/)
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('sends multipart to /v1/tools/ocr and returns page text', async () => {
  const received: { purpose: string; auth: string; filePart: boolean } = { purpose: '', auth: '', filePart: false }
  const server = createServer((req, res) => {
    const chunks: Buffer[] = []
    req.on('data', (chunk: Buffer) => chunks.push(chunk))
    req.on('end', () => {
      const body = Buffer.concat(chunks).toString('utf8')
      received.auth = String(req.headers.authorization ?? '')
      received.purpose = /name="purpose"\r\n\r\n([^\r]+)/.exec(body)?.[1] ?? ''
      received.filePart = /name="file"; filename="board\.png"/.test(body)
      res.setHeader('content-type', 'application/json')
      res.end(JSON.stringify({ id: 'ocr-local', page_count: 1, pages: [{ index: 0, text: 'СЧЁТ 12:7', finish_reason: 'stop' }], usage: { total_tokens: 7 } }))
    })
  })
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  const port = (server.address() as AddressInfo).port
  const dir = await mkdtemp(join(tmpdir(), 'gputool-ocr-'))
  try {
    const file = join(dir, 'board.png')
    await writeFile(file, 'fake-image-bytes')
    const result = await runDeepseekOcr({ apiKey: 'test-key', baseUrl: `http://127.0.0.1:${port}`, file, startPercent: 0 })
    assert.equal(result.pages[0].text, 'СЧЁТ 12:7')
    assert.equal(result.id, 'ocr-local')
    assert.equal(received.purpose, 'text_recognition')
    assert.equal(received.auth, 'Bearer test-key')
    assert.ok(received.filePart)
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()))
    await rm(dir, { recursive: true, force: true })
  }
})

test('the key is required and never echoed back', async () => {
  await assert.rejects(runDeepseekOcr({ apiKey: '  ', file: '/nonexistent.png' }), /not configured/)
  const server = createServer((req, res) => {
    req.resume()
    req.on('end', () => {
      res.statusCode = 401
      res.end('{"error":"invalid api key test-key-123"}')
    })
  })
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  const port = (server.address() as AddressInfo).port
  const dir = await mkdtemp(join(tmpdir(), 'gputool-ocr-'))
  try {
    const file = join(dir, 'board.png')
    await writeFile(file, 'x')
    await assert.rejects(runDeepseekOcr({ apiKey: 'test-key-123', baseUrl: `http://127.0.0.1:${port}`, file }), (error: any) => {
      assert.equal(error.code, 'auth-failed')
      assert.ok(!String(error.message).includes('test-key-123'), 'ключ не должен попадать в текст ошибки')
      return true
    })
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()))
    await rm(dir, { recursive: true, force: true })
  }
})
