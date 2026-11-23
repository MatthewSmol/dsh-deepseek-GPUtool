# GPU tool

[English](README.en.md) | Русский

DSH-плагин и локальный MCP-сервис для общей GPU-работы: ffmpeg NVDEC/NVENC,
ONNX Runtime CUDA (YuNet/SFace), OCR (DeepSeek → PP-OCRv5), YOLO11 и GPU-скрипты. У каждой задачи
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
- **ffmpeg на GPU**: NVDEC (`-hwaccel cuda`, `*_cuvid`), zero-copy `scale_cuda`
  и NVENC (`h264/hevc/av1`, пресеты `p1–p7`, `cq`, `-tag:v hvc1`) из сборки
  `ffmpeg-nvenc`. Операции: **encode, decode, remux, trim, split, concat, frames,
  thumb, raw**; автоматический выбор кодека (NVENC → CPU-фолбэк).
- **OCR с приоритетом DeepSeek**: по умолчанию `engine="auto"` сначала вызывает
  **DeepSeek Tools API** (`POST /v1/tools/ocr`, `text_recognition`), ключ берётся
  из credential-store DSH (`DEEPSEEK_API_KEY`) и в конфиг не пишется; при
  недоступности ключа/сети — локальный **PP-OCRv5** на CUDA. `engine="local"`
  принудительно локальный путь, `engine="deepseek"` — только API.
- **Скан лиц YuNet + SFace**: постоянный Rust-воркер (`bin/gputool-face`) держит
  сессии ONNX Runtime между заданиями и считает кадры батчами; Python-воркер
  остаётся запасным путём (на замере Rust тёплый быстрее в 7,7×).
- **ONNX Runtime CUDA**: универсальный `onnx`, скан лиц **YuNet + SFace** и
  локальный OCR **PP-OCRv5** (детектор + распознавание) через
  `CUDAExecutionProvider`, без тихого перехода на CPU при `requireGpu`.
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
- **Доктор возможностей**: `/dev/dxg`, nvidia-smi, NVENC/CUVID, `scale_cuda`,
  ONNX Runtime и cuDNN, torch CUDA, ultralytics, модели YOLO/OCR/лиц — с точными
  командами исправления.
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
- Поддержка Vulkan (`libplacebo` + `fruc_vulkan`) удалена: в WSL без NVIDIA-ICD
  устройство поднималось на программном llvmpipe, интерполяция падала на
  отсутствии `VK_NV_optical_flow`. Остались CUDA-пути NVDEC/NVENC/`scale_cuda`.
- `LD_LIBRARY_PATH` строился по зашитому `python3.12`; теперь site-packages
  находится через glob — cuDNN из venv подхватывается.
- TensorRT в списке провайдеров падал без библиотек — по умолчанию CUDA → CPU.
- OCR: 3-канальный вход, склейка соседних боксов строки, вертикальная ориентация,
  корректная индексация словаря PP-OCRv5.
- Панель: перевод вынесен в состояние (`tr`) — переключение языка работает без
  перезагрузки.

**Известные ограничения**
- Словарь `ppocrv5_dict.txt` в `ocr-ru` не кириллический (851 символ) — нужен
  корректный словарь PP-OCRv5 для чтения русского табло.
- Браузерная панель визуально не проверялась (GUI требует token-URL).

### 1.0.1 — 2026-11-23

- Удалена поддержка Vulkan: фильтры `libplacebo`/`fruc_vulkan`, параметр
  `vulkan` в `ffmpeg`, требование GPU для него и три проверки доктора
  (`vulkan-device`, `vulkan-filters`, `vulkaninfo`). Причина: на хосте нет
  GPU-ICD, цепочка работала на llvmpipe, а интерполяция недоступна без
  `VK_NV_optical_flow`. Интерфейс: русский по умолчанию + английский (RU/EN).
- README: русская и [английская](README.en.md) версии с общей историей.

### 1.1.0 — 2026-11-23

- **OCR через DeepSeek Tools API** (`POST /v1/tools/ocr`) с приоритетом
  «DeepSeek → локальный PP-OCRv5»: `engine: auto|deepseek|local`, `purpose:
  text_recognition|rich_layout_recognition`, результат — `ocr.jsonl` по страницам.
- Ключ `DEEPSEEK_API_KEY` берётся из credential-store DSH и передаётся
  MCP-серверу только через окружение дочернего процесса; ошибки редактируются
  (`Bearer …`, `sk-…`, сам ключ).
- Проверено вживую: `SCORE 11`, `:score: 3 : 2` распознаны на GPU-хосте за ~3 с,
  токены и время пишутся в хвост лога; 401/сеть → автоматический фолбэк на
  локальный OCR.
- Новые настройки: `ocrEngine`, `deepseekBaseUrl`, `deepseekApiKeyEnv`,
  `deepseekTimeoutMs` (170 с — лимит API 180 с).
- Тесты: +5 (разбор ответа, multipart на локальный сервер, отсутствие ключа в
  тексте ошибки, лимиты файла) — всего 17.

### 1.2.2 — 2026-11-23

- Вкладка **«Задачи»** показывает только актуальную работу: задания со статусом
  `running` и `queued`; завершённые остаются в истории.
- Вкладка **«История»** показывает **последние 10 записей** (после поиска и
  фильтра) с подписью; детали по-прежнему открываются кликом.

### 1.2.1 — 2026-11-23

- **Исправлено: вкладка «Задачи» падала** — компоненту прогресса не передавался
  переводчик (`tr`), из-за чего любой список заданий ломал рендер.
- **Настройки показывают движки**: какой OCR выбран (`auto`/`deepseek`/`local`),
  найден ли ключ DeepSeek и из какой переменной, где лежат локальные модели
  PP-OCRv5, область табло, режим воркера лиц и доступность Rust-воркера.
- В `/api/state` добавлен блок `capabilities` (OCR + лица), чтобы панель
  показывала фактическое состояние, а не догадки.

### 1.2.0 — 2026-11-23

- **Rust-воркер лиц** (`rust/face-worker`, бинарь `bin/gputool-face`): постоянный
  процесс с одним `onnxruntime` Environment на всё время работы. Сессии YuNet и
  SFace создаются один раз, кадры читаются из ffmpeg, готовятся по предвычисленным
  весам ресайза и считаются батчами (до 8 кадров), NMS и выравнивание — на Rust.
- **Замер на одном клипе** (20 с, 200 кадров, 3 лица в кадре, с эталоном):
  Python-воркер — **24,2 с** (121 мс/кадр), Rust тёплый — **3,14 с** (15,7 мс/кадр),
  Rust холодный — 4,28 с. Ускорение **~7,7×**; без эталона Rust — 11,2 мс/кадр.
- **Почему**: Python тратит время на старт ORT-сессий в каждом задании, на
  `estimateAffinePartial2D`/`warpAffine` на каждое лицо и на GIL; Rust держит
  сессии, а ресайз с таблицами весов дал 18,5 → 11,2 мс/кадр.
- **Честность**: воркер сам определяет, прикрепился ли CUDA EP, и сообщает
  реальный `device` в событии `ready` и в результате; при недоступности Rust или
  его падении движок автоматически откатывается на Python-путь.
- **Настройки**: `faceWorker: auto|rust|python` (по умолчанию `auto`), MCP-параметры
  `worker`, `batch`, `threads`; сборка `npm run build:rust`, тесты `npm run test:rust`.
- **Паритет**: боксы совпадают с Python на 83 % при IoU ≥ 0,5 (медиана 0,893);
  различие — округление билинейной интерполяции, лица находятся те же.
- Тесты: +2 Node (доступность бинаря и поиск `libonnxruntime.so`) и 3 Rust
  (`nms`, `similarity`, `cosine`).

### Планы

- Rust-воркеры для внутренних стадий (детекция, OCR-постобработка, трекинг) —
  см. `AGENTS.md`.
- Словарь PP-OCRv5 и профиль области табло по умолчанию.
- Прямой стрим прогресса в панель вместо опроса раз в секунду.
