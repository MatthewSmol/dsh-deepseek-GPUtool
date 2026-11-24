/** Прогресс python-заданий: собственный протокол и ffmpeg-строки. */
import assert from 'node:assert/strict'
import test from 'node:test'
import { parseProgressLine, timecodeToMs } from '../engine/progress.js'

test('explicit progress percentage wins', () => {
  assert.deepEqual(parseProgressLine('progress: 42%'), { percent: 42 })
  assert.deepEqual(parseProgressLine('progress 7 %'), { percent: 7 })
})

test('ffmpeg out_time_us with a known duration gives percent and clamp', () => {
  const parsed = parseProgressLine('out_time_us=1250000', 5_000)
  assert.equal(parsed?.processedMs, 1250)
  assert.equal(parsed?.percent, 25)
  assert.equal(parsed?.totalMs, 5_000)
  const over = parseProgressLine('out_time_us=99999999', 1_000)
  assert.equal(over?.percent, 99.9)
})

test('out_time_ms is microseconds as ffmpeg emits it', () => {
  const parsed = parseProgressLine('out_time_ms=2500000', 10_000)
  assert.equal(parsed?.processedMs, 2500)
  assert.equal(parsed?.percent, 25)
})

test('time= timecodes and frame/fps/speed are parsed', () => {
  assert.equal(timecodeToMs('00:00:12.34'), 12_340)
  assert.equal(timecodeToMs('01:02:03.5'), 3_723_500)
  assert.equal(timecodeToMs('nope'), null)
  const timed = parseProgressLine('time=00:00:12.34')
  assert.equal(timed?.processedMs, 12_340)
  const counters = parseProgressLine('frame= 120 fps=25.0 speed=1.5x')
  assert.equal(counters?.frame, 120)
  assert.equal(counters?.fps, 25)
  assert.equal(counters?.speed, 1.5)
})

test('eta is computed from speed and content is ignored without numbers', () => {
  const parsed = parseProgressLine('frame=10 fps=25 speed=2.0x out_time_ms=2000000', 10_000)
  assert.equal(parsed?.percent, 20)
  assert.equal(parsed?.etaMs, 4000)
  assert.equal(parseProgressLine('START'), null)
  assert.equal(parseProgressLine('hello world'), null)
})
