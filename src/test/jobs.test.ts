/**
 * Очередь заданий: watchdog мёртвых процессов, подбор зависших и жёсткая отмена.
 * Тесты используют короткие таймауты и временный GPUTOOL_HOME, без GPU.
 */
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { JobManager } from '../engine/jobs.js'
import type { SubmitInput } from '../engine/jobs.js'
import type { DeviceRef, JobExecutor } from '../engine/types.js'

const DEVICE: DeviceRef = { kind: 'gpu', index: 0, name: 'test-gpu' }

function makeManager(): JobManager {
  return new JobManager({
    historyLimit: 20,
    jobHistoryLimit: 20,
    gpuConcurrency: 1,
    cpuConcurrency: 1,
    gpuAvailable: () => true,
    stallTimeoutMs: 120,
    pidDeadGraceMs: 60,
    cancelGraceMs: 60,
    logger: { info: () => undefined, warn: () => undefined },
  })
}

function input(operation: string): SubmitInput {
  return {
    kind: 'ffmpeg',
    operation,
    params: {},
    device: DEVICE,
    engine: 'test',
    argv: [],
    tool: 'test',
    argsHash: operation,
    label: null,
    clientName: 'test',
    who: { source: 'system', sessionId: null, label: null, client: 'test' },
  }
}

const never: JobExecutor = async () => await new Promise<void>(() => undefined)

test('watchdog fails a stalled job and releases the queue', async () => {
  const home = mkdtempSync(join(tmpdir(), 'gputool-jobs-'))
  const previous = process.env.GPUTOOL_HOME
  process.env.GPUTOOL_HOME = home
  try {
    const manager = makeManager()
    const stuck = manager.submit(input('stuck'), never)
    const next = manager.submit(input('next'), async (ctx) => {
      ctx.log('next started')
    })
    await new Promise((resolve) => setTimeout(resolve, 200))
    const reaped = await manager.reap()
    const failed = manager.get(stuck.id)
    const done = manager.get(next.id)
    assert.ok(reaped.failed.includes(stuck.id) || failed?.status === 'failed', `stuck status=${failed?.status} reaped=${JSON.stringify(reaped)}`)
    assert.equal(failed?.error?.code, 'stalled')
    await new Promise((resolve) => setTimeout(resolve, 100))
    assert.equal(manager.get(next.id)?.status, 'succeeded', `next status=${manager.get(next.id)?.status}`)
    assert.equal(done?.status === 'succeeded' || manager.get(next.id)?.status === 'succeeded', true)
  } finally {
    if (previous === undefined) delete process.env.GPUTOOL_HOME
    else process.env.GPUTOOL_HOME = previous
    rmSync(home, { recursive: true, force: true })
  }
})

test('watchdog fails a job whose tracked pid is dead', async () => {
  const home = mkdtempSync(join(tmpdir(), 'gputool-jobs-'))
  const previous = process.env.GPUTOOL_HOME
  process.env.GPUTOOL_HOME = home
  const child = spawn('sleep', ['30'])
  try {
    const manager = makeManager()
    const job = manager.submit(input('dead-pid'), async (ctx) => {
      if (child.pid != null) ctx.trackPid(child.pid)
      await new Promise<void>(() => undefined)
    })
    await new Promise((resolve) => setTimeout(resolve, 100))
    child.kill('SIGKILL')
    await new Promise((resolve) => setTimeout(resolve, 120))
    await manager.reap()
    await new Promise((resolve) => setTimeout(resolve, 100))
    await manager.reap()
    const record = manager.get(job.id)
    assert.equal(record?.status, 'failed')
    assert.equal(record?.error?.code, 'process-died')
  } finally {
    child.kill('SIGKILL')
    if (previous === undefined) delete process.env.GPUTOOL_HOME
    else process.env.GPUTOOL_HOME = previous
    rmSync(home, { recursive: true, force: true })
  }
})

test('cancel force-finalizes a running job and frees the slot', async () => {
  const home = mkdtempSync(join(tmpdir(), 'gputool-jobs-'))
  const previous = process.env.GPUTOOL_HOME
  process.env.GPUTOOL_HOME = home
  try {
    const manager = makeManager()
    const job = manager.submit(input('cancel-me'), never)
    await new Promise((resolve) => setTimeout(resolve, 20))
    assert.equal(manager.cancel(job.id), true)
    await new Promise((resolve) => setTimeout(resolve, 150))
    const record = manager.get(job.id)
    assert.equal(record?.status, 'cancelled')
    assert.equal(manager.queueInfo().running, 0)
  } finally {
    if (previous === undefined) delete process.env.GPUTOOL_HOME
    else process.env.GPUTOOL_HOME = previous
    rmSync(home, { recursive: true, force: true })
  }
})
