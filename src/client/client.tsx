/**
 * Браузерная половина плагина GPU tool.
 *
 * Регистрирует пункт в левом сайдбаре и главную панель, передавая ей функцию
 * перевода и минифицированный CSS из сборки Vite. Язык интерфейса: русский по
 * умолчанию, английский — второй; переключатель находится в шапке панели и
 * сохраняется в localStorage.
 */
import css from './styles.css?inline'
import { createPanel, type Language } from './panel.js'
import { en, ru } from './locale.js'

declare global {
  interface Window {
    __ModuleLoader__: {
      load: (definition: { id: string; factory: (require: (id: string) => unknown) => unknown }) => void
    }
  }
}

const NS = 'gputool'
const PANEL_ID = 'gputool'
const LANG_KEY = 'gputool.lang'
const DEFAULT_LANGUAGE: Language = 'ru'

function storedLanguage(): Language {
  try {
    const value = typeof localStorage !== 'undefined' ? localStorage.getItem(LANG_KEY) : null
    return value === 'en' || value === 'ru' ? value : DEFAULT_LANGUAGE
  } catch {
    return DEFAULT_LANGUAGE
  }
}

function persistLanguage(value: Language): void {
  try {
    if (typeof localStorage !== 'undefined') localStorage.setItem(LANG_KEY, value)
  } catch {
    /* localStorage недоступен — остаёмся на языке по умолчанию */
  }
}

window.__ModuleLoader__.load({
  id: '@local/dsh-gputool',
  factory(require) {
    const React = require('react') as any

    function Icon(props: { size?: number; active?: boolean }): any {
      const size = props?.size ?? 16
      const active = Boolean(props?.active)
      return React.createElement(
        'svg',
        {
          viewBox: '0 0 64 64',
          width: size,
          height: size,
          'aria-hidden': true,
          style: { display: 'block', color: active ? 'var(--dsw-alias-brand-primary)' : 'currentColor' },
        },
        React.createElement('rect', { x: 6, y: 12, width: 52, height: 34, rx: 6, fill: 'none', stroke: 'currentColor', strokeWidth: 4 }),
        React.createElement('path', { d: 'M6 20h52M6 38h52', stroke: 'currentColor', strokeWidth: 2.5, opacity: 0.65 }),
        React.createElement('path', { d: 'M22 56h20M32 46v10', stroke: 'currentColor', strokeWidth: 4, strokeLinecap: 'round' }),
        React.createElement('circle', { cx: 47, cy: 29, r: 4.5, fill: '#47b26b' }),
        React.createElement('path', { d: 'M16 29h14M16 35h9', stroke: '#e0a33e', strokeWidth: 3, strokeLinecap: 'round' }),
      )
    }

    function apply(ctx: any): void {
      ctx.effect?.(() => ctx.locale.register(NS, { en, ru }), 'gputool: словари')
      const bound = typeof ctx.locale?.bind === 'function' ? ctx.locale.bind(NS) : null
      // Русский — язык по умолчанию: сначала таблица ru, английский — запасной.
      const tr = (lang: Language, key: string): string => {
        const table = lang === 'en' ? en : ru
        if (typeof table[key] === 'string') return table[key]
        try {
          const value = typeof bound === 'function' ? bound(key) : undefined
          return typeof value === 'string' && value ? value : (en[key] ?? key)
        } catch {
          return en[key] ?? key
        }
      }

      const Panel = createPanel({
        React,
        t: tr,
        css,
        language: storedLanguage,
        onLanguageChange: persistLanguage,
      })

      let layout: any = null
      try {
        ctx.inject?.(['layout'], (scoped: any) => {
          layout = scoped.layout
        })
      } catch {
        layout = null
      }

      ctx.slots.inject('main', () => {
        const offPanel = ctx.slots.register(
          { name: 'main', key: PANEL_ID, locale: NS, inject: () => ({ t: (key: string) => tr(storedLanguage(), key) }) },
          Panel,
        )
        return () => {
          try {
            if (layout?.panelInfo?.getSnapshot?.().activePanelId === PANEL_ID) layout.selectPanel(null)
          } catch {
            /* игнорируем */
          }
          offPanel?.()
        }
      })

      ctx.slots.inject('sidebar.panellist', () =>
        ctx.slots.register(
          {
            name: 'sidebar.panellist',
            id: PANEL_ID,
            order: 40,
            locale: NS,
            label: () => tr(storedLanguage(), 'nav'),
          },
          Icon,
        ),
      )
    }

    return { inject: ['slots', 'locale'], apply }
  },
})
