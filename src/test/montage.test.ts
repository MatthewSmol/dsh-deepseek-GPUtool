/**
 * Планировщик монтажа: один граф на concat + апскейл + fps + счёт + голос.
 * Тесты проверяют argv, содержимое filter_complex и честные фолбэки.
 */
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { buildFfmpegPlan, buildMontagePlan, inferRawOutput, rawDurationFromArgs, type FfmpegParams, type MontageSceneInfo } from '../engine/ffmpeg.js'
import { defaultSettings } from '../engine/settings.js'
import type { FfmpegCaps } from '../engine/types.js'

const FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

function caps(filters: string[], encoders = ['hevc_nvenc', 'h264_nvenc', 'libx264', 'aac']): FfmpegCaps {
  return { path: '/ffmpeg', version: 'test', encoders, decoders: [], filters, hwaccels: ['cuda'] }
}

const FULL = caps(['concat', 'drawtext', 'amix', 'adelay', 'minterpolate', 'libplacebo', 'fruc_vulkan', 'scale'])

function scenes(count = 2): MontageSceneInfo[] {
  return Array.from({ length: count }, (_, index) => ({
    path: `/video/scene-${index}.mp4`,
    durationSec: 6,
    width: 1280,
    height: 720,
    fps: 30,
    hasAudio: index === 0,
  }))
}

function params(overrides: Partial<FfmpegParams['montage']> = {}): FfmpegParams {
  return {
    operation: 'montage',
    codec: 'hevc',
    quality: 'balanced',
    output: '/out/final.mp4',
    overwrite: true,
    montage: {
      scenes: [
        { input: '/video/scene-0.mp4', startSec: 2, durationSec: 6 },
        { input: '/video/scene-1.mp4', durationSec: 6 },
      ],
      upscale: { width: 1920, height: 1080 },
      fps: 60,
      score: {
        events: [
          { atSec: 1, untilSec: 5, text: 'Счёт 1 : 0' },
          { atSec: 6, untilSec: 10, text: 'Счёт 1 : 1' },
        ],
      },
      audio: { tracks: [{ path: '/audio/voice.ogg', atSec: 2, gainDb: -2 }] },
      ...overrides,
    },
  }
}

const context = { vulkanHardware: false, vulkanAvailable: true, fontFile: FONT, tempDir: '/tmp/montage-job' }

test('montage builds one filter graph with concat, EWA, minterpolate, score and voice', () => {
  const plan = buildMontagePlan(params(), FULL, defaultSettings(), scenes(), 1, context)
  const graph = String(plan.argv[plan.argv.indexOf('-filter_complex') + 1])
  assert.match(graph, /concat=n=2:v=1:a=1\[vc\]\[ac\]/)
  assert.match(graph, /libplacebo=w=1920:h=1080:upscaler=ewa_lanczossharp/)
  assert.match(graph, /minterpolate=fps=60/)
  assert.match(graph, /drawtext=fontfile=.*DejaVuSans-Bold\.ttf/)
  assert.equal((graph.match(/enable='between\(t,/g) ?? []).length, 2)
  assert.match(graph, /adelay=2000:all=1/)
  assert.match(graph, /amix=inputs=2:duration=first:dropout_transition=0:normalize=0/)
  assert.match(graph, /loudnorm=I=-16/)
  assert.equal(plan.mode.upscaler, 'ewa')
  assert.equal(plan.mode.interpolator, 'minterpolate')
  assert.equal(plan.usedEncoder, 'hevc_nvenc')
  assert.equal(plan.totalMs, 12_000)
  assert.ok(plan.argv.includes('-init_hw_device'))
  assert.ok(plan.argv.includes('hevc_nvenc'))
  assert.ok(plan.argv.includes('-filter_complex'))
  const scoreFiles = plan.tempFiles.filter((file) => file.path.includes('score-'))
  assert.deepEqual(scoreFiles.map((file) => file.content), ['Счёт 1 : 0', 'Счёт 1 : 1'])
})

test('montage falls back to lanczos and duplicated frames with warnings when filters are missing', () => {
  const basic = caps(['concat', 'drawtext', 'amix', 'adelay', 'scale'])
  const plan = buildMontagePlan(params({ score: undefined, audio: undefined }), basic, defaultSettings(), scenes(), 0, context)
  const graph = String(plan.argv[plan.argv.indexOf('-filter_complex') + 1])
  assert.match(graph, /scale=1920:1080:flags=lanczos/)
  assert.match(graph, /fps=60/)
  assert.doesNotMatch(graph, /libplacebo|minterpolate/)
  assert.equal(plan.mode.upscaler, 'lanczos')
  assert.equal(plan.mode.interpolator, 'duplicate')
  assert.ok(plan.warnings.some((warning) => warning.includes('EWA is replaced')))
  assert.ok(plan.warnings.some((warning) => warning.includes('minterpolate')))
})

test('strict mode rejects unavailable EWA and fruc instead of silently falling back', () => {
  const basic = caps(['concat', 'drawtext', 'scale'])
  assert.throws(
    () => buildMontagePlan(params({ strict: true, upscale: { width: 1920, height: 1080, method: 'ewa' } }), basic, defaultSettings(), scenes(), 0, context),
    (error: any) => error.code === 'montage-unsupported',
  )
  assert.throws(
    () => buildMontagePlan(params({ strict: true, interpolate: 'fruc', score: undefined, audio: undefined }), FULL, defaultSettings(), scenes(), 0, context),
    (error: any) => error.code === 'montage-unsupported',
  )
})

test('fast profile uses lanczos and frame duplication', () => {
  const plan = buildMontagePlan(params({ profile: 'fast', score: undefined, audio: undefined }), FULL, defaultSettings(), scenes(), 0, context)
  const graph = String(plan.argv[plan.argv.indexOf('-filter_complex') + 1])
  assert.match(graph, /scale=1920:1080:flags=lanczos/)
  assert.match(graph, /fps=60/)
  assert.equal(plan.mode.interpolator, 'duplicate')
})

test('montage validates scene count and duration before running ffmpeg', () => {
  const many = Array.from({ length: 41 }, (_, index) => ({ ...scenes(1)[0], path: `/v/${index}.mp4` }))
  assert.throws(() => buildMontagePlan(params(), FULL, defaultSettings(), many, 0, context), /at most 40 scenes/)
  const zero = scenes(2).map((scene) => ({ ...scene, durationSec: 0 }))
  assert.throws(() => buildMontagePlan(params({ scenes: [{ input: '/v/a.mp4' }, { input: '/v/b.mp4' }] }), FULL, defaultSettings(), zero, 0, context), /no usable duration/)
})

test('dropping scene audio without voice tracks creates silence of the full length', () => {
  const plan = buildMontagePlan(params({ score: undefined, audio: { sceneAudio: 'drop' } }), FULL, defaultSettings(), scenes(), 0, context)
  const graph = String(plan.argv[plan.argv.indexOf('-filter_complex') + 1])
  assert.match(graph, /anullsrc=r=48000:cl=stereo,atrim=duration=12\.000\[aout\]/)
})


test('raw keeps user argv, injects progress once and derives the duration', () => {
  const args = ['-i', '/video/a.mp4', '-i', '/audio/voice.m4a', '-filter_complex', '[0:v]fps=60[v]', '-map', '[v]', '-map', '1:a', '-c:v', 'h264_nvenc', '-c:a', 'aac', '-t', '12.5', '-y', '/out/final.mp4']
  const plan = buildFfmpegPlan({ operation: 'raw', args, inputs: ['/video/extra.mp4'], progress: true }, FULL, defaultSettings(), null, 0)
  assert.deepEqual(plan.argv.slice(0, 2), ['-hide_banner', '-nostdin'])
  assert.equal(plan.argv.filter((item) => item === '-progress').length, 1)
  assert.equal(plan.argv.filter((item) => item === '-y').length, 1)
  assert.equal(plan.totalMs, 12_500)
  assert.ok(plan.argv.includes('/out/final.mp4'))
  // inputs-сахар идёт до пользовательских аргументов, чтобы индексы были предсказуемы
  const sugar = plan.argv.indexOf('/video/extra.mp4')
  assert.ok(sugar > 0 && sugar < plan.argv.indexOf('-filter_complex'))
  assert.equal(plan.usedEncoder, 'h264_nvenc')
})

test('raw without args or inputs is rejected and helpers find the output path', () => {
  assert.throws(() => buildFfmpegPlan({ operation: 'raw', args: [] }, FULL, defaultSettings(), null, 0), /args or inputs is required/)
  assert.equal(rawDurationFromArgs(['-ss', '2', '-t', '5']), 5)
  assert.equal(rawDurationFromArgs(['-ss', '2', '-to', '9.5']), 7.5)
  assert.equal(inferRawOutput(['-i', 'a.mp4', '-c:v', 'libx264', '/out/x.mp4']), '/out/x.mp4')
  assert.equal(inferRawOutput(['-f', 'null', '-']), null)
})

test('montage adds subtitles and PNG overlays with correct input order and audio indices', () => {
  const dir = mkdtempSync(join(tmpdir(), 'gputool-montage-'))
  const plate = join(dir, 'plate.png')
  const subs = join(dir, 'fragments.ru.vtt')
  writeFileSync(plate, Buffer.from('89504e470d0a1a0a', 'hex'))
  writeFileSync(subs, 'WEBVTT\n')
  const paramsWithLayers: FfmpegParams = {
    ...params(),
    montage: {
      ...params().montage!,
      subtitles: { path: subs, fontsDir: '/usr/share/fonts/truetype/dejavu' },
      overlays: [{ path: plate, x: 24, y: 32, width: 320, height: 80, opacity: 0.9, fromSec: 2, toSec: 10 }],
    },
  }
  const plan = buildMontagePlan(paramsWithLayers, FULL, defaultSettings(), scenes(), 0, context)
  const graph = String(plan.argv[plan.argv.indexOf('-filter_complex') + 1])
  assert.match(graph, /subtitles=filename='.*fragments\.ru\.vtt':fontsdir='\/usr\/share\/fonts\/truetype\/dejavu'\[vsub\]/)
  assert.match(graph, /\[2:v\]format=rgba,scale=320:80,colorchannelmixer=aa=0\.900\[ov0\]/)
  assert.match(graph, /overlay=24:32:shortest=0:eof_action=repeat:enable='between\(t,2\.000,10\.000\)'/)
  assert.match(graph, /format=yuv420p\[vout\]/)
  // сцены 0..1, оверлей 2, голос 3
  assert.match(graph, /\[3:a\]aresample=48000/)
  assert.ok(plan.argv.includes('-loop') && plan.argv.includes('1'))
  const audioAt = plan.argv.indexOf('/audio/voice.ogg')
  const overlayAt = plan.argv.indexOf(plate)
  assert.ok(overlayAt > 0 && overlayAt < audioAt)
  rmSync(dir, { recursive: true, force: true })
})

test('montage rejects missing overlay files before ffmpeg starts', () => {
  const broken: FfmpegParams = {
    ...params(),
    montage: { ...params().montage!, overlays: [{ path: '/video/does-not-exist.png', x: 0, y: 0 }] },
  }
  assert.throws(() => buildMontagePlan(broken, FULL, defaultSettings(), scenes(), 0, context), (error: any) => error.code === 'overlay-missing')
})

// --- CQ/CRF и окно извлечения кадров -----------------------------------------

function cqValues(argv: string[], flag: string): number[] {
  const values: number[] = []
  for (let index = 0; index < argv.length - 1; index += 1) {
    if (argv[index] === flag) values.push(Number(argv[index + 1]))
  }
  return values
}

test('encode accepts an explicit cq and passes it to NVENC', () => {
  const plan = buildFfmpegPlan(
    { operation: 'encode', input: '/in.mp4', output: '/out.mp4', codec: 'hevc', quality: 'quality', cq: 17, overwrite: true },
    caps(['scale'], ['hevc_nvenc', 'libx264', 'aac']),
    defaultSettings(),
    null,
    0,
  )
  assert.ok(plan.argv.includes('hevc_nvenc'))
  assert.deepEqual(cqValues(plan.argv, '-cq'), [17], plan.argv.join(' '))
})

test('encode rejects cq outside 0..51', () => {
  assert.throws(
    () =>
      buildFfmpegPlan(
        { operation: 'encode', input: '/in.mp4', output: '/out.mp4', cq: 60 },
        caps(['scale']),
        defaultSettings(),
        null,
        0,
      ),
    /cq must be between 0 and 51/,
  )
})

test('frames honours startSec and durationSec', () => {
  const plan = buildFfmpegPlan(
    { operation: 'frames', input: '/in.mp4', outputDir: '/frames', startSec: 300, durationSec: 10, fps: 1, overwrite: true },
    caps(['scale']),
    defaultSettings(),
    null,
    0,
  )
  const start = plan.argv.indexOf('-ss')
  const input = plan.argv.indexOf('-i')
  const end = plan.argv.indexOf('-t')
  assert.ok(start >= 0 && plan.argv[start + 1] === '300', plan.argv.join(' '))
  assert.ok(input > start, 'seek must come before the input')
  assert.ok(end > input && plan.argv[end + 1] === '10', plan.argv.join(' '))
})
