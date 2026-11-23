import { defineConfig } from 'vite'
import { resolve } from 'node:path'

/**
 * Браузерная сборка плагина GPU tool. DSH загружает `dist/client.js` как
 * классический скрипт, вызывающий `window.__ModuleLoader__.load`; React берётся
 * из таблицы модулей загрузчика через `require('react')`, поэтому его нельзя
 * резолвить из node_modules. CSS импортируется через `?inline` и попадает
 * строкой в бандл, отдельный файл стилей не нужен.
 */
export default defineConfig({
  build: {
    lib: {
      entry: resolve(process.cwd(), 'src/client/client.tsx'),
      formats: ['iife'],
      name: 'GpuToolClient',
      fileName: () => 'client.raw.js',
    },
    outDir: 'dist/vite-client',
    emptyOutDir: true,
    target: 'es2022',
    minify: 'esbuild',
    cssCodeSplit: false,
    sourcemap: false,
    reportCompressedSize: true,
  },
  define: {
    'process.env.NODE_ENV': JSON.stringify('production'),
  },
  optimizeDeps: { noDiscovery: true },
})
