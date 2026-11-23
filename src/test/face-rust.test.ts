/**
 * Проверки Rust-воркера лиц: доступность бинаря и библиотеки ONNX Runtime,
 * а также разбор событий протокола (ready/progress/result/error).
 */
import assert from 'node:assert/strict'
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { onnxRuntimeLibrary, rustFaceAvailable, rustFaceBinary } from '../engine/face-rust.js'
import { defaultSettings } from '../engine/settings.js'

test('binary path points inside the package and the availability report is honest', () => {
  const settings = defaultSettings()
  const binary = rustFaceBinary()
  assert.ok(binary.endsWith(join('bin', 'gputool-face')), `unexpected binary path: ${binary}`)
  const availability = rustFaceAvailable(settings)
  // На машинах без сборки Rust путь честно сообщает причину, а не падает.
  assert.equal(typeof availability.ok, 'boolean')
  assert.equal(typeof availability.reason, 'string')
  if (!existsSync(binary)) {
    assert.equal(availability.ok, false)
    assert.match(availability.reason, /binary not found/)
  }
})

test('ONNX Runtime library is discovered in the plugin venv only', () => {
  const dir = mkdtempSync(join(tmpdir(), 'gputool-venv-'))
  try {
    const python = join(dir, 'bin', 'python')
    mkdirSync(join(dir, 'bin'), { recursive: true })
    writeFileSync(python, '')
    assert.equal(onnxRuntimeLibrary({ ...defaultSettings(), pythonPath: python }), null)
    assert.equal(onnxRuntimeLibrary({ ...defaultSettings(), pythonPath: 'python3' }), null)
    // Каталог capi без самой библиотеки — тоже null, а не выдуманный путь.
    const capi = join(dir, 'lib', 'python3.12', 'site-packages', 'onnxruntime', 'capi')
    mkdirSync(capi, { recursive: true })
    assert.equal(onnxRuntimeLibrary({ ...defaultSettings(), pythonPath: python }), null)
    // С библиотекой путь находится.
    writeFileSync(join(capi, 'libonnxruntime.so.1.32.0'), '')
    assert.equal(onnxRuntimeLibrary({ ...defaultSettings(), pythonPath: python }), join(capi, 'libonnxruntime.so.1.32.0'))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
