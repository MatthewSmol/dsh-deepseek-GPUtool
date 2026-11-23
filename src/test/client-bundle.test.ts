import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'

test('браузерный бандл регистрирует пункт сайдбара и главную панель', () => {
  const code = readFileSync('dist/client.js', 'utf8')
  assert.ok(code.includes('__ModuleLoader__'))
  assert.ok(!/^\s*(?:import|export)\s/m.test(code))
  let definition: any = null
  const sandbox: any = {
    window: {
      __ModuleLoader__: {
        load(value: unknown) {
          definition = value
        },
      },
    },
    document: { visibilityState: 'visible', addEventListener() {}, removeEventListener() {} },
    navigator: { clipboard: { writeText: async () => undefined } },
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    console,
  }
  vm.runInNewContext(code, sandbox)
  assert.ok(definition, 'бандл должен вызвать __ModuleLoader__.load')
  assert.equal(definition.id, '@local/dsh-gputool')

  const ReactStub = {
    createElement: () => null,
    useState: (value: unknown) => [value, () => undefined],
    useEffect: () => undefined,
  }
  const module = definition.factory((id: string) => {
    if (id === 'react') return ReactStub
    throw new Error(`неожиданный модуль: ${id}`)
  })
  assert.deepEqual([...module.inject].sort(), ['locale', 'slots'])

  const injections: string[] = []
  const registrations: Array<{ name: string; id?: string; key?: string }> = []
  const ctx = {
    effect(fn: () => unknown) {
      fn()
    },
    inject() {},
    locale: {
      register() {},
      bind: () => (key: string) => key,
    },
    slots: {
      inject(name: string, callback: () => unknown) {
        injections.push(name)
        callback()
      },
      register(options: { name: string; id?: string; key?: string }) {
        registrations.push({ name: options.name, id: options.id, key: options.key })
        return () => undefined
      },
    },
  }
  module.apply(ctx)
  assert.deepEqual(injections.sort(), ['main', 'sidebar.panellist'])
  assert.ok(registrations.some((entry) => entry.name === 'main' && entry.key === 'gputool'))
  assert.ok(registrations.some((entry) => entry.name === 'sidebar.panellist' && entry.id === 'gputool'))
})
