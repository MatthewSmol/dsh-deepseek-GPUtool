# GPU tool

DSH-плагин и локальный MCP-сервис для общей GPU-работы: ffmpeg NVDEC/NVENC/Vulkan,
ONNX Runtime CUDA (YuNet/SFace), PP-OCRv5, YOLO11 и GPU-скрипты Python. У каждой
задачи есть живой прогресс (процент, ETA, fps), а история вызовов хранит, какая
сессия её запустила и сколько времени это заняло.

- MCP-инструменты: `mcp__gputool__{status,probe,ffmpeg,yolo,onnx,face,ocr,python,jobs,job,cancel,history,gpu,setup,settings}`
- Панель: левый сайдбар → GPU tool (Обзор / Задачи / История / Аренда / Настройки)
- HTTP API: `/gputool/api/*` (loopback, защищён токеном)
- Установка окружения: `mcp__gputool__setup` → `install-runtime` (torch CUDA, onnxruntime-gpu, ultralytics, lap)

Подробности — в [docs/DETAIL.md](docs/DETAIL.md).

```bash
npm install --cache ./.npm-cache
npm run build
npm test
```

Затем установите каталог через менеджер плагинов DSH.
