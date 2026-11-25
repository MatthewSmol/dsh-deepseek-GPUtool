/**
 * Трекинг мяча: путь к модели, доступность и валидация параметров.
 * Инференс не запускается — только CPU-логика; ONNX проверяется e2e.
 */
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, join } from 'node:path'
import test from 'node:test'
import { ballAvailable, ballModelPath, buildBallConfig, parseBallResult } from '../engine/ball-rust.js'
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

    // Фон для bgMode=concat: путь проверяется, число кадров ограничено.
    const withBackground = buildBallConfig(
      { bgMode: 'concat', background: source, backgroundSamples: 16, frames: 8 },
      settings,
      source,
      join(dir, 'out'),
      'auto',
    )
    assert.equal(withBackground.bgMode, 'concat')
    assert.equal(withBackground.background, source)
    assert.equal(withBackground.backgroundSamples, 16)
    assert.equal(withBackground.frames, 8)
    assert.throws(() => buildBallConfig({ background: join(dir, 'missing.png') }, settings, source, join(dir, 'out'), 'auto'), /Background not found/)
    assert.throws(() => buildBallConfig({ backgroundSamples: 4 }, settings, source, join(dir, 'out'), 'auto'), /backgroundSamples must be between 8 and 128/)
    assert.throws(() => buildBallConfig({ backgroundSamples: 500 }, settings, source, join(dir, 'out'), 'auto'), /backgroundSamples must be between 8 and 128/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})


test('ball image result keeps coordinates and annotation', () => {
  const parsed = parseBallResult({
    engine: 'rust',
    device: 'cuda:0',
    providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'],
    images: 1,
    width: 1280,
    height: 720,
    x: 210.99,
    y: 180.21,
    confidence: 0.2546,
    visible: true,
    annotated: '/tmp/frame_ball.png',
    model: { frames: 3, channels: 9, width: 512, height: 288, bgMode: '' },
    elapsedMs: 775,
  })
  assert.equal(parsed.visible, true)
  assert.ok(Math.abs((parsed.x ?? 0) - 210.99) < 0.01)
  assert.ok(Math.abs((parsed.confidence ?? 0) - 0.2546) < 1e-3)
  assert.equal(parsed.annotated, '/tmp/frame_ball.png')
  assert.deepEqual(parsed.model, { frames: 3, width: 512, height: 288, offset: 0 })

  // Видео-результат без координат не должен получать фантомные значения.
  const video = parseBallResult({ frames: 125, detected: 30, jsonl: '/tmp/ball.jsonl' })
  assert.equal(video.visible, false)
  assert.equal(video.x, undefined)
  assert.equal(video.annotated, null)
  assert.equal(video.frames, 125)

  // Фон, использованный на видео, приходит в результат как строка.
  const withBackground = parseBallResult({ frames: 10, background: 'median of 32 frames' })
  assert.equal(withBackground.background, 'median of 32 frames')
  assert.equal(parseBallResult({ frames: 10 }).background, null)
})
