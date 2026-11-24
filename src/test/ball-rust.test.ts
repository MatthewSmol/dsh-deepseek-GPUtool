/**
 * Трекинг мяча: путь к модели, доступность и валидация параметров.
 * Инференс не запускается — только CPU-логика; ONNX проверяется e2e.
 */
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, join } from 'node:path'
import test from 'node:test'
import { ballAvailable, ballModelPath, buildBallConfig } from '../engine/ball-rust.js'
import { defaultSettings } from '../engine/settings.js'

function fixture(): { dir: string; source: string; model: string } {
  const dir = mkdtempSync(join(tmpdir(), 'gputool-ball-'))
  const source = join(dir, 'clip.mp4')
  const model = join(dir, 'tracknetv2.onnx')
  writeFileSync(source, 'not-a-real-video')
  writeFileSync(model, 'not-a-real-onnx')
  return { dir, source, model }
}

test('ball model path falls back to modelsDir/ball', () => {
  const settings = { ...defaultSettings(), ballModelPath: '' }
  const fallback = ballModelPath(settings)
  assert.ok(['blurball_tt.onnx', 'tracknetv3_badminton.onnx', 'tracknetv2.onnx'].includes(basename(fallback)), fallback)
  assert.equal(ballModelPath(settings, '/m/custom.onnx'), '/m/custom.onnx')
  assert.equal(ballModelPath({ ...settings, ballModelPath: '/m/preset.onnx' }), '/m/preset.onnx')
})

test('ball availability names the missing piece', () => {
  const { dir, model } = fixture()
  try {
    const availability = ballAvailable({ ...defaultSettings(), ballModelPath: model })
    assert.equal(typeof availability.ok, 'boolean')
    if (!availability.ok) assert.match(availability.reason, /binary not found|libonnxruntime|model not found/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('ball config takes defaults from settings and rejects invalid numbers', () => {
  const { dir, source, model } = fixture()
  try {
    const settings = { ...defaultSettings(), ballModelPath: model, ballThreshold: 0.4, ballMaxGapFrames: 8, ballAnnotate: true }
    const config = buildBallConfig({}, settings, source, join(dir, 'out'), 'auto', 1)
    assert.equal(config.threshold, 0.4)
    assert.equal(config.maxGapFrames, 8)
    assert.equal(config.annotate, true)
    assert.equal(config.gpuIndex, 1)
    assert.equal(config.device, 'auto')
    assert.throws(() => buildBallConfig({ threshold: 2 }, settings, source, join(dir, 'out'), 'auto'), /threshold must be between 0 and 1/)
    assert.throws(() => buildBallConfig({ fps: 0 }, settings, source, join(dir, 'out'), 'auto'), /fps must be between 1 and 60/)
    assert.throws(() => buildBallConfig({ maxGapFrames: 500 }, settings, source, join(dir, 'out'), 'auto'), /maxGapFrames must be between 0 and 120/)
    assert.throws(() => buildBallConfig({}, settings, join(dir, 'missing.mp4'), join(dir, 'out'), 'auto'), /Source not found/)
    assert.throws(() => buildBallConfig({ model: join(dir, 'missing.onnx') }, settings, source, join(dir, 'out'), 'auto'), /Model not found/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
