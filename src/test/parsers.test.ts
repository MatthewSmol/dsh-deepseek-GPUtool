import test from 'node:test'
import assert from 'node:assert/strict'
import { parseGpuCsv, parseProcessCsv } from '../engine/gpu.js'
import { applyProgressLine, parseListing, buildFfmpegPlan } from '../engine/ffmpeg.js'
import { defaultSettings } from '../engine/settings.js'
import type { FfmpegCaps } from '../engine/types.js'

test('parses nvidia-smi CSV with a comma in the GPU name', () => {
  const csv = [
    '0, GPU-aaa, NVIDIA GeForce RTX 3070, 8192, 1024, 7168, 620.70.03, 8.6',
    '1, GPU-bbb, NVIDIA GeForce RTX 3060 Ti, 8192, 2048, 6144, 620.70.03, 8.6',
  ].join('\n')
  const gpus = parseGpuCsv(csv)
  assert.equal(gpus.length, 2)
  assert.equal(gpus[0].name, 'NVIDIA GeForce RTX 3070')
  assert.equal(gpus[0].freeMB, 7168)
  assert.equal(gpus[1].index, 1)
  assert.equal(gpus[1].uuid, 'GPU-bbb')
})

test('keeps commas inside a GPU name', () => {
  const gpus = parseGpuCsv('0,GPU-x,NVIDIA Test, Card, 8192, 0, 8192, 1.0, 8.0')
  assert.equal(gpus.length, 1)
  assert.equal(gpus[0].name, 'NVIDIA Test, Card')
})

test('parses compute processes', () => {
  const processes = parseProcessCsv('GPU-aaa, 4242, python3, 512\nGPU-bbb, 7, ffmpeg, 64')
  assert.equal(processes.length, 2)
  assert.equal(processes[1].pid, 7)
  assert.equal(processes[1].usedMB, 64)
})

test('parses ffmpeg listing lines', () => {
  const encoders = parseListing(' V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)\n V..... libx264              libx264 (codec h264)', 'encoders')
  assert.deepEqual(encoders, ['h264_nvenc', 'libx264'])
  const filters = parseListing(' TS cas               V->V       Contrast Adaptive Sharpen.\n T.. scale_cuda          V->V       CUDA hardware-accelerated scaling filter', 'filters')
  assert.deepEqual(filters, ['cas', 'scale_cuda'])
})

test('progress lines use microseconds and percent math stays stable', () => {
  const state = { outTimeUs: null, frame: null, fps: null, speed: null, totalSize: null, done: false }
  applyProgressLine(state, 'out_time_us=2500000')
  applyProgressLine(state, 'out_time_ms=2500000')
  assert.equal(state.outTimeUs, 2500000)
  applyProgressLine(state, 'frame=75')
  applyProgressLine(state, 'fps=30.5')
  applyProgressLine(state, 'speed=1.8x')
  applyProgressLine(state, 'progress=end')
  assert.equal(state.frame, 75)
  assert.equal(state.fps, 30.5)
  assert.equal(state.speed, 1.8)
  assert.equal(state.done, true)
})

const caps: FfmpegCaps = {
  path: '/usr/bin/ffmpeg',
  version: 'ffmpeg test',
  encoders: ['h264_nvenc', 'hevc_nvenc', 'libx264', 'libx265'],
  decoders: ['hevc_cuvid', 'h264_cuvid'],
  filters: ['cas', 'scale_cuda'],
  hwaccels: ['cuda', 'cuvid'],
}

test('plans a GPU encode with NVENC and hvc1 tag', () => {
  const plan = buildFfmpegPlan({ operation: 'encode', input: '/in.mp4', output: '/out.mp4', codec: 'hevc', quality: 'quality' }, caps, defaultSettings(), { durationSec: 10, format: null, sizeBytes: null, video: null, audio: null }, 0)
  assert.ok(plan.argv.includes('hevc_nvenc'))
  assert.ok(plan.argv.includes('hvc1'))
  assert.ok(plan.argv.includes('-progress'))
  assert.equal(plan.totalMs, 10000)
  assert.equal(plan.binary, '/usr/bin/ffmpeg')
})

test('falls back to a CPU encoder and records the warning', () => {
  const settings = { ...defaultSettings(), allowCpuFallback: true }
  const plan = buildFfmpegPlan({ operation: 'encode', input: '/in.mp4', output: '/out.mp4', codec: 'h264' }, { ...caps, encoders: ['libx264'] }, settings, null, 0)
  assert.ok(plan.argv.includes('libx264'))
  assert.ok(plan.warnings.some((warning) => warning.includes('libx264')))
})


test('split plans a stream-copy segmenter and rejects missing patterns', () => {
  const plan = buildFfmpegPlan({ operation: 'split', input: '/in.mp4', output: '/out/seg-%03d.mp4', segmentSec: 4 }, caps, defaultSettings(), { durationSec: 12, format: null, sizeBytes: null, video: null, audio: null }, 0)
  assert.ok(plan.argv.includes('segment'))
  assert.ok(plan.argv.includes('copy'))
  assert.throws(() => buildFfmpegPlan({ operation: 'split', input: '/in.mp4', output: '/out/seg.mp4', segmentSec: 4 }, caps, defaultSettings(), null, 0), /pattern/)
})
