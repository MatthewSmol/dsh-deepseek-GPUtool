/** Разбор device: numeric-string приводится к числу, мусор даёт понятную ошибку. */
import assert from 'node:assert/strict'
import test from 'node:test'
import { normalizeDeviceParam } from '../engine/engine.js'

test('device accepts auto, cpu, numbers and numeric strings', () => {
  assert.equal(normalizeDeviceParam(undefined), undefined)
  assert.equal(normalizeDeviceParam('auto'), 'auto')
  assert.equal(normalizeDeviceParam('cpu'), 'cpu')
  assert.equal(normalizeDeviceParam(1), 1)
  assert.equal(normalizeDeviceParam('1'), 1)
  assert.equal(normalizeDeviceParam(' 0 '), 0)
})

test('device rejects bad values with an actionable message', () => {
  assert.throws(() => normalizeDeviceParam('gpu1'), /device must be "auto", "cpu" or an integer GPU index/)
  assert.throws(() => normalizeDeviceParam(-1), /non-negative integer/)
  assert.throws(() => normalizeDeviceParam(1.5), /non-negative integer/)
})
