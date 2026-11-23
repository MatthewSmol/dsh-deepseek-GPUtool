import test from 'node:test'
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { readFileSync, rmSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import readline from 'node:readline'
import { VERSION } from '../engine/engine.js'

const TOOLS = ['status', 'probe', 'ffmpeg', 'yolo', 'onnx', 'face', 'upscale', 'ocr', 'python', 'jobs', 'job', 'cancel', 'history', 'gpu', 'setup', 'settings']

test('MCP server negotiates, lists tools/resources and answers status', async () => {
  const home = mkdtempSync(join(tmpdir(), 'gputool-mcp-'))
  const child = spawn(process.execPath, ['dist/mcp/index.js', '--stdio'], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env, GPUTOOL_HOME: home },
    cwd: process.cwd(),
  })
  const lines = readline.createInterface({ input: child.stdout })
  const pending = new Map<number, (value: any) => void>()
  lines.on('line', (line) => {
    try {
      const message = JSON.parse(line)
      if (typeof message.id === 'number' && pending.has(message.id)) {
        const resolve = pending.get(message.id)!
        pending.delete(message.id)
        resolve(message)
      }
    } catch {
      /* не-JSON игнорируем */
    }
  })
  let nextId = 1
  const request = (method: string, params: unknown): Promise<any> =>
    new Promise((resolve, reject) => {
      const id = nextId++
      pending.set(id, resolve)
      child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
      setTimeout(() => reject(new Error(`timeout ${method}`)), 20_000)
    })
  try {
    const init = await request('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'mcp-test', version: '1' } })
    assert.equal(init.result.serverInfo.name, 'gputool')
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`)
    const tools = await request('tools/list', {})
    const names = tools.result.tools.map((tool: any) => tool.name).sort()
    assert.deepEqual(names, [...TOOLS].sort())
    const resources = await request('resources/list', {})
    assert.ok(resources.result.resources.some((entry: any) => entry.uri === 'gputool://doctor'))
    const status = await request('tools/call', { name: 'status', arguments: {} })
    const value = JSON.parse(status.result.content[0].text)
    assert.equal(value.version, VERSION)
    assert.equal(typeof value.apiBaseUrl, 'string')
    await new Promise((resolve) => setTimeout(resolve, 300))
    const endpoint = JSON.parse(readFileSync(join(home, 'endpoint.json'), 'utf8'))
    assert.ok(endpoint.port > 0)
    assert.ok(endpoint.token.length >= 32)
  } finally {
    child.kill('SIGTERM')
    rmSync(home, { recursive: true, force: true })
  }
})
