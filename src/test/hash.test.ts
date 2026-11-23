import test from 'node:test'
import assert from 'node:assert/strict'
import { stableStringify } from '../engine/engine.js'
import { argsHash } from '../index.js'

test('stable stringify sorts object keys recursively', () => {
  assert.equal(stableStringify({ b: 1, a: [{ d: 2, c: 3 }] }), '{"a":[{"c":3,"d":2}],"b":1}')
})

test('plugin hint hash matches the engine hashing order', () => {
  const args = { source: '/x.mp4', fps: 1, region: { w: 10, h: 20, x: 1, y: 2 } }
  const fromString = stableStringify(args)
  assert.equal(argsHash(args), argsHash({ region: { y: 2, x: 1, h: 20, w: 10 }, fps: 1, source: '/x.mp4' }))
  assert.ok(fromString.includes('"h":20'))
})
