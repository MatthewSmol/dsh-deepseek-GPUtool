# GPU tool

[English](README.en.md) | Русский

DSH-плагин и локальный MCP-сервис для общей GPU-работы: ffmpeg NVDEC/NVENC/Vulkan,
ONNX Runtime CUDA (YuNet/SFace), PP-OCRv5, YOLO11 и GPU-скрипты. У каждой задачи
есть живой прогресс (процент, ETA, fps), а история вызовов хранит, какая сессия её
запустила и сколько времени это заняло.

- MCP-инструменты: `mcp__gputool__{status,probe,ffmpeg,yolo,onnx,face,ocr,python,jobs,job,cancel,history,gpu,setup,settings}`
- Панель: левый сайдбар → GPU tool (Обзор / Задачи / История / Аренда / Настройки)
- Язык интерфейса: **русский по умолчанию**, английский — переключателем RU / EN в шапке панели
- HTTP API: `/gputool/api/*` (loopback, защищён токеном)
- Установка окружения: `mcp__gputool__setup` → `install-runtime` (torch CUDA, onnxruntime-gpu, ultralytics, lap)

Подробности — в [docs/DETAIL.md](docs/DETAIL.md).

```bash
npm install --cache ./.npm-cache
npm run build
npm test
```

Затем установите каталог через менеджер плагинов DSH.

---

## История изменений

### 1.0.0 — 2026-11-23

Первый рабочий релиз: MCP-сервис GPU-работы, панель в сайдбаре, живой прогресс и
история вызовов.

**Добавлено**

- **MCP-сервер** (`src/mcp`) на TypeScript: stdio-транспорт и loopback HTTP API в
  одном процессе. Инструменты: `status`, `probe`, `ffmpeg`, `yolo`, `onnx`, `face`,
  `ocr`, `python`, `jobs`, `job`, `cancel`, `history`, `gpu`, `setup`, `settings`;
  ресурсы `gputool://doctor|gpus|jobs|jobs/{id}|history`.
- **ffmpeg на GPU**: NVDEC (`-hwaccel cuda`, `*_cuvid`), zero-copy `scale_cuda`,
  NVENC (`h264/hevc/av1`, пресеты `p1–p7`, `cq`, `-tag:v hvc1`), Vulkan-путь
  `libplacebo` (EWA Lanczos + `cas`) и `fruc_vulkan` (60 fps) из сборки
  `ffmpeg-nvenc`. Операции: **encode, decode, remux, trim, split, concat, frames,
  thumb, raw**; автоматический выбор кодека (NVENC → CPU-фолбэк).
- **ONNX Runtime CUDA**: универсальный `onnx`, скан лиц **YuNet + SFace** и OCR
  **PP-OCRv5** (детектор + распознавание) — всё через `CUDAExecutionProvider`,
  без тихого перехода на CPU при `requireGpu`.
- **YOLO11** на CUDA: `predict` и трекинг **BoT-SORT** (`track=true`,
  `tracker=botsort.yaml`), JSONL с боксами, `trackId` и центром объекта.
- **Python-воркер** для произвольных пайплайнов: `CUDA_VISIBLE_DEVICES`,
  `LD_LIBRARY_PATH` из venv (torch и пакеты `nvidia-*`), зеркалирование строк
  `progress: N%` в прогресс задания.
- **Очередь и аренда**: одна задача на GPU, две на CPU, приоритет по свободной
  VRAM, аренда устройства (lease) для внешних процессов, отмена, восстановление
  после рестарта с кодом `interrupted`.
- **Живой прогресс**: процент, фаза, `processed/total`, fps, ETA по средней
  скорости, счётчик кадров и флагов `stalled`.
- **История**: кто (id сессии + метка + источник `tool|http|mcp`), длительность,
  статус, устройство, выходные файлы и хвост лога; фильтры по сессии/типу/операции.
- **Панель** (React + Vite IIFE с минификацией): вкладки «Обзор», «Задачи»,
  «История», «Аренда», «Настройки»; матрица возможностей доктора с копируемыми
  рецептами; переключатель языка **RU / EN** (по умолчанию русский, выбор
  сохраняется в `localStorage`).
- **Доктор возможностей**: `/dev/dxg`, nvidia-smi, NVENC/CUVID, Vulkan-устройство
  (с распознаванием программного llvmpipe), фильтры Vulkan, ONNX Runtime и cuDNN,
  torch CUDA, ultralytics, модели YOLO/OCR/лиц — с точными командами исправления.
- **setup**: изолированный venv `$DSH_HOME/gputool/venv` (uv, Python 3.12) —
  torch с CUDA-индекса, cuDNN, `onnxruntime-gpu`, `ultralytics`, `lap`,
  `opencv-python-headless`, `numpy`; проверка `torch.cuda.is_available()`.
- **DSH-плагин** (`src/index.ts`): монтирует `@deepseek-ai/dsh-mcp-client`,
  резолвит его от установки DSH, проксирует `/gputool` на loopback API и
  привязывает вызов к сессии через `tools/pre-execute`.
- **Тесты**: 13 тестов `node --test` — парсеры nvidia-smi/ffmpeg/прогресса,
  сборка argv, стабильный хеш подсказок, MCP-рукопожатие и контракт бандла.

**Исправлено по ходу разработки**

- Двойной путь к бинарю ffmpeg в argv (`binary в argv` → `cwd ffmpeg`).
- `CUDA_VISIBLE_DEVICES` для ffmpeg ломал `-hwaccel_device`; теперь устройство
  задаётся только `-gpu`/`-hwaccel_device`.
- `out_time_ms` в ffmpeg — микросекунды: процент и ETA больше не «прыгают».
- Vulkan: `upscaler=ewa_lanczos4sharp` недоступен → `ewa_lanczossharp`, а
  `fruc_vulkan` принимает `fps=`; цепочка собрана как
  `format=nv12,hwupload,libplacebo,…,hwdownload,format=nv12`.
- `LD_LIBRARY_PATH` строился по зашитому `python3.12`; теперь site-packages
  находится через glob — cuDNN из venv подхватывается.
- TensorRT в списке провайдеров падал без библиотек — по умолчанию CUDA → CPU.
- OCR: 3-канальный вход, склейка соседних боксов строки, вертикальная ориентация,
  корректная индексация словаря PP-OCRv5.
- Панель: перевод вынесен в состояние (`tr`) — переключение языка работает без
  перезагрузки.

**Известные ограничения**

- Vulkan в WSL без NVIDIA-ICD поднимается на llvmpipe; `fruc_vulkan` требует
  расширение `VK_NV_optical_flow` и на программном устройстве недоступен.
- Словарь `ppocrv5_dict.txt` в `ocr-ru` не кириллический (851 символ) — нужен
  корректный словарь PP-OCRv5 для чтения русского табло.
- Браузерная панель визуально не проверялась (GUI требует token-URL).

### Планы

- Rust-воркеры для внутренних стадий (детекция, OCR-постобработка, трекинг) —
  см. `AGENTS.md`.
- Словарь PP-OCRv5 и профиль области табло по умолчанию.
- Прямой стрим прогресса в панель вместо опроса раз в секунду.
