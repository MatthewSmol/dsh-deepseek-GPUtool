/**
 * GPU-апскейл x2 (SPAN fp16): доступность и валидация конфига.
 * Инференс здесь не запускается — только CPU-логика, GPU проверяется e2e.
 */
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import {
  buildUpscaleConfig,
  isImageSource,
  upscaleAvailable,
  upscaleModelPath,
} from '../engine/upscale-rust.js'
import { defaultSettings } from '../engine/settings.js'

function fixture(): { dir: string; source: string; model: string } {
  const dir = mkdtempSync(join(tmpdir(), 'gputool-upscale-'))
  const source = join(dir, 'frame.png')
  const model = join(dir, 'span2x_ch48_fp16.onnx')
  writeFileSync(source, 'not-a-real-png')
  writeFileSync(model, 'not-a-real-onnx')
  return { dir, source, model }
}

test('image and video sources are detected by extension', () => {
  assert.equal(isImageSource('/x/frame.PNG'), true)
  assert.equal(isImageSource('/x/frame.webp'), true)
  assert.equal(isImageSource('/x/clip.mp4'), false)
  assert.equal(isImageSource('/x/noext'), false)
})

test('upscale model path falls back to modelsDir/upscale', () => {
  const settings = { ...defaultSettings(), upscaleModelPath: '' }
  assert.equal(upscaleModelPath(settings), join(settings.modelsDir, 'upscale', 'span2x_ch48_fp16.onnx'))
  assert.equal(upscaleModelPath(settings, '/models/custom.onnx'), '/models/custom.onnx')
  const explicit = { ...settings, upscaleModelPath: '/models/preset.onnx' }
  assert.equal(upscaleModelPath(explicit), '/models/preset.onnx')
})

test('availability names the missing piece instead of guessing', () => {
  const { dir, model } = fixture()
  try {
    const settings = { ...defaultSettings(), upscaleModelPath: model }
    const unavailable = upscaleAvailable(settings)
    // Бинарь в пакете пока может отсутствовать — причина должна быть строкой.
    assert.equal(typeof unavailable.ok, 'boolean')
    assert.equal(typeof unavailable.reason, 'string')
    if (!unavailable.ok) assert.match(unavailable.reason, /binary not found|libonnxruntime|model not found/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('config validation rejects batch outside the allowed range', () => {
  const { dir, source, model } = fixture()
  try {
    const settings = { ...defaultSettings(), upscaleModelPath: model, upscaleBatch: 8 }
    assert.throws(() => buildUpscaleConfig({ batch: 0 }, settings, source, join(dir, 'out.png')), /batch must be between 1 and 64/)
    assert.throws(() => buildUpscaleConfig({ batch: 65 }, settings, source, join(dir, 'out.png')), /batch must be between 1 and 64/)
    assert.throws(() => buildUpscaleConfig({}, settings, join(dir, 'missing.png'), join(dir, 'out.png')), /Source not found/)
    assert.throws(() => buildUpscaleConfig({ model: join(dir, 'missing.onnx') }, settings, source, join(dir, 'out.png')), /Model not found/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('valid config carries batch, codec and both ffmpeg paths', () => {
  const { dir, source, model } = fixture()
  try {
    const settings = { ...defaultSettings(), upscaleModelPath: model, upscaleBatch: 4 }
    const config = buildUpscaleConfig({}, settings, source, join(dir, 'out.png'))
    assert.equal(config.batch, 4)
    assert.equal(config.codec, 'hevc')
    assert.equal(config.overwrite, false)
    assert.equal(config.ffmpegPath, settings.ffmpegGpuPath)
    assert.equal(config.ffprobePath, settings.ffprobeGpuPath)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
