# GPU tool

[English](README.en.md) | Русский

DSH-плагин и локальный MCP-сервис для общей GPU-работы: ffmpeg NVDEC/NVENC,
ONNX Runtime CUDA (YuNet/SFace, BlurBall, SPAN x2), OCR (DeepSeek → PP-OCRv5),
YOLO11, трекинг мяча и GPU-скрипты. У каждой задачи есть живой прогресс
(процент, ETA, fps), а история вызовов хранит, какая сессия её запустила и
сколько времени это заняло.

- MCP-инструменты: `mcp__gputool__{status,probe,ffmpeg,yolo,onnx,face,ocr,ball,upscale,python,jobs,job,cancel,history,gpu,setup,settings}`
- Панель: левый сайдбар → GPU tool (Обзор / Задачи / История / Аренда / Настройки)
- Язык интерфейса: **русский по умолчанию**, английский — переключателем RU / EN в шапке панели
- HTTP API: `/gputool/api/*` (loopback, защищён токеном)
- Апскейл **x2 (SPAN fp16)** строго на CUDA: батчи, NVDEC/NVENC, без тайлинга и без CPU-фолбэка
- Трекинг мяча настольного тенниса: **BlurBall** (ONNX, CUDA) → координаты и видео с точкой
- Модели на диске: `$DSH_HOME/gputool/models/ball/blurball_tt.onnx` (мяч) и
  `$DSH_HOME/gputool/models/upscale/span2x_ch48_fp16.onnx` (апскейл x2); путь
  меняется в настройках (`ballModelPath`, `upscaleModelPath`)
- Установка окружения: `mcp__gputool__setup` → `install-runtime` (torch CUDA, onnxruntime-gpu, ultralytics, lap)

Подробности — в [docs/DETAIL.md](docs/DETAIL.md).

```bash
npm install --cache ./.npm-cache
npm run build        # TypeScript + клиентский бандл
npm run build:rust   # воркеры face / upscale / ball (cargo)
npm test             # Node-тесты
npm run test:rust    # Rust-тесты воркеров
```

Затем установите каталог через менеджер плагинов DSH.

---

## Админ-API

- `POST /api/admin/reap` — снять зависшие задания (мёртвый pid/нет прогресса) и
  разблокировать очередь.
- `POST /api/admin/restart` — перезапустить MCP-процесс сервиса (DSH поднимет
  его снова). Зависшие running-задания при старте помечаются `interrupted`.
- `GET /api/state` → блок `ffmpeg` (NVENC/фильтры/hwaccels) и `job.warnings`
  c предупреждениями плана.

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

### 1.6.3 — 2026-11-24

- **Прогресс python-заданий**: кроме `progress: NN%` парсятся ffmpeg-строки
  `out_time_us=`/`out_time_ms=`/`time=`, `frame=`, `fps=`, `speed=`; параметр
  `durationSec` включает проценты и ETA.
- Пустой процент больше не выглядит «сломанным»: в панели видно живое время
  выполнения и подсказку «скрипт не сообщает прогресс».
- Причина отсутствия процентов у `final2.py`: скрипт запускает ffmpeg через
  `capture_output=True` и печатает только `START` — gputool не видит `-progress`.
  Решение: гнать энкод через `ffmpeg raw` (он сам добавляет `-progress pipe:1`)
  или пробрасывать строки ffmpeg из скрипта.
- Тесты: 59 Node (+5 для парсера прогресса).

### 1.6.2 — 2026-11-24

- **`vf` больше не теряется**: параметр `vf` (полная цепочка фильтров) работает в
  `encode`, `trim`, `split`, `frames`, `thumb`; `scale` склеивается с ним через
  запятую. Причина бага: `args` учитывался только в `raw`, а `encode` молча
  игнорировал `-vf` — теперь `args` вне `raw` даёт ошибку `args-raw-only`.
- **GPU-фильтры**: для программных фильтров (`scale`, `unsharp`, `crop`…)
  NVDEC-декод отключается автоматически (предупреждение в job.warnings),
  энкодинг остаётся NVENC. Чисто CUDA-цепочки (`scale_cuda`, `libplacebo`)
  сохраняют `-hwaccel cuda`.
- **`encode`** теперь учитывает `startSec`/`durationSec` (`-ss` до входа,
  `-t` после), а `codec:"copy"` с фильтрами падает с `filters-with-copy`
  вместо тихой потери фильтра.
- **`frames`** применяет `scale`/`vf` вместе с `fps` (`fps=1,scale=640:360`).
- **`/api/state.ffmpeg`**: бинарь, доступность NVENC/CPU-энкодеров и фильтров
  (`scale_cuda`, `hwdownload`, `unsharp`, `cas`, `subtitles`, `libplacebo`),
  hwaccels; `job.warnings` отдаётся в `/api/jobs` и панели.
- Проверено на GPU: 3-секундный клип 1280×720 → `vf:"scale=640:360"` дал
  640×360, h264_nvenc, длительность 3.000 с.
- Тесты: 54 Node (+8: vf/copy/seek/frames/diagnostics).

### 1.6.1 — 2026-11-24

- **Watchdog очереди**: pid дочерних процессов регистрируются (`ctx.trackPid`),
  мёртвый процесс → `failed/process-died`, отсутствие прогресса → `stalled`,
  таймаут → `timeout`; слот освобождается, очередь продолжает работу.
- **Отмена running** теперь жёсткая: SIGTERM→SIGKILL по pid, через 5 с запись
  финализируется принудительно (`cancelled`), даже если executor завис.
- **Админ-API**: `POST /api/admin/reap` (снять зависшие) и
  `POST /api/admin/restart` (перезапуск MCP-процесса); в панели — кнопка
  «Снять зависшие» и бейдж «зависло».
- **Состояние**: `/api/state` и `/api/jobs` показывают `alive`, `stalled`,
  `pids`; `queue` дополнен `stalled`/`aliveRunning`.
- **ffmpeg encode**: параметр `cq` (0–51) переопределяет `-cq`/`-crf` пресета.
- **ffmpeg frames**: учитывает `startSec`/`durationSec` (`-ss` до входа, `-t`
  после) — «10 секунд с 300-й» больше не извлекает с начала.
- **upscale**: `width`/`height` — точное целевое разрешение (пре-ресайз в
  target/2, затем x2), например 1280×720 → 1280×720 без двойного прохода.
- **GPU**: `usedByProcessesMB`/`otherMB` — видно, сколько VRAM занято вне
  namespace (другой контейнер/хост); auto-выбор пишет выбранную GPU и free,
  избегает карт с <1 ГБ свободных.
- **device**: numeric-string (`"1"`) принимается как индекс; мусор даёт
  сообщение `device must be "auto", "cpu" or an integer GPU index`.

### 1.6.0 — 2026-11-24

- **Апскейл теперь только x2** (4x-UltraSharp удалён): лёгкая **SPAN x2 fp16**
  вместо RRDBNet, инференс строго на CUDA без CPU-фолбэка.
- **fp16 + батчи**: кадры идут пакетами (`batch`, по умолчанию 8), один
  `session.run` на батч; тайлинг удалён (`tile=0`), при попытке тайла — ошибка.
- **Все веса скачаны**: SPAN-архив (ch48/ch52) и Real-ESRGAN x2plus; экспорт
  `scripts/export_span2x_onnx.py` → `span2x_ch48_fp16.onnx` (835 КБ, sha256).
- **Качество fp16**: PSNR против PyTorch fp32 — **57.99 дБ**, MAE 0.26 уровня.
- **Скорость** на RTX 3070 (512×288→1024×576, batch 8): **32.3 fps**;
  инференс 24.1 мс/кадр, PNG-декод 1.8 мс, PNG-энкод 5.0 мс.
- **Видео-путь**: NVDEC → батч CUDA → NVENC, 3 с клип 512×288→1024×576 за
  2.8 с (~27 fps), HEVC; аудио копируется при наличии.
- **Чанк c00**: 2000 кадров за **61.8 с**; ETA на все 13 частей (25 925
  кадров) — **~13.4 мин** (старый 4x: 231 мс/кадр ≈ 4.3 fps ≈ 100 мин).
- Настройки: `upscaleModelPath` (дефолт `span2x_ch48_fp16.onnx`), `upscaleBatch`;
  `upscaleTile`/`upscaleOverlap` удалены; doctor проверяет fp16-сессию и factor 2.

### 1.5.2 — 2026-11-24

- **BlurBall как основная модель детекции мяча** (настольный теннис, MIT):
  веса лежат в репозитории `mrzkp/blurball` (`best_model` — SE-вариант, F1
  видео-инференса **0.959** по метаданным чекпойнта), экспорт в ONNX —
  `scripts/export_blurball_onnx.py` (авто-определение SE, вход `[1,9,288,512]`,
  выход `[1,3,288,512]`).
- Нормализация ImageNet теперь включается автоматически и по имени `blurball`
  (раньше только `wasb`); sigmoid не дублируется — BlurBall уже отдаёт
  вероятности.
- На реальном клипе 1280×720@25 (20 с): **488/502 кадров** с мячом,
  **324 реальные детекции** (было 274 у WASB), средняя уверенность **0.794**,
  14,5 с вместе с разметкой; точка проверена пиксельно (conf 0.965).
- README: BlurBall — рекомендуемый путь, WASB остаётся альтернативой;
  версия 1.5.2.

### 1.5.1 — 2026-11-24

- **Скачаны реальные веса WASB-SBDT** (`wasb_tennis_best.pth.tar`, 6,1 МБ, MIT)
  и экспортированы в ONNX скриптом `scripts/export_wasb_onnx.py`
  (HRNet, вход `[1,9,288,512]`, выход `[1,3,288,512]`, opset 18).
- Воркер `ball` научился: выбирать канал из нескольких heatmap (`outputIndex`,
  по умолчанию последний), автоматически применять **sigmoid** к логитам и
  **ImageNet-нормализацию** для моделей WASB (`normalize: auto|imagenet|none`),
  отбрасывать телепорты (`maxJumpPx`, по умолчанию 30 % ширины).
- Проверено на реальном клипе 1280×720@25 (20 с, WASB): **456/502 кадров**
  с мячом (274 реальные детекции, средняя уверенность 0.78), медианный шаг
  9,4 px, время обработки ~13,7 с (включая разметку), `ball.mp4` HEVC.
- Точка на видео проверена пиксельно: в кадре реальной детекции (conf 0.94)
  29 жёлтых пикселей ровно в координатах модели, в оригинале — 0.
- README: команды скачивания и экспорта, sha256, нормализация и новые параметры;
  DETAIL обновлён.

### 1.5.0 — 2026-11-24

- **Трекинг мяча настольного тенниса** (`mcp__gputool__ball`): ONNX-модели
  TrackNetV2/V3 и WASB (3 кадра → heatmap) через новый Rust-воркер
  `bin/gputool-ball`; CUDA при наличии, иначе честный CPU-фолбэк для `auto`.
- Пайплайн: ffmpeg → кольцевой буфер кадров → билинейный ресайз 512×288 →
  тензор `[1, frames*3, H, W]` → heatmap → субпиксельный центр (argmax + центроид),
  отбрасывание телепортов, заполнение пропусков до `maxGapFrames`, медианное
  сглаживание.
- Выход: `ball.jsonl` (frame/time/visible/x/y/conf), `ball.csv`, `ball.ass` и
  видео с точкой (`subtitles`-фильтр + NVENC, аудио copy).
- Настройки: `ballModelPath`, `ballThreshold`, `ballMaxGapFrames`,
  `ballAnnotate`, `ballDevice`; поля в панели, проверки в doctor
  (`ball-worker`, `ball-model`).
- Модель не коммитится: в README — экспорт в ONNX
  (`opset 18`, `[1,9,288,512]`) для TrackNetV2/V3 и WASB.
- Тесты: +3 Rust (heatmap/постобработка/ASS) и +3 Node (валидация) — 37 Node,
  8 Rust.

### 1.4.0 — 2026-11-24

- **GPU-апскейл 4x-UltraSharp** (`mcp__gputool__upscale`): отдельный Rust-воркер
  `bin/gputool-upscale`, инференс **строго на CUDA** — сессия создаётся с
  `session.disable_cpu_ep_fallback`, поэтому модель либо целиком на GPU, либо
  задание падает с `cuda-ep-unavailable`/`gpu-unsupported` (CPU-фолбэка нет).
- **Картинки**: тайлинг с перекрытием (по умолчанию 256/16), линейный
  feather-блендинг без швов, RGBA сохраняет прозрачность (альфа — Lanczos).
- **Видео**: NVDEC-декод → кадры → CUDA-инференс → NVENC (`hevc_nvenc`/`h264_nvenc`),
  аудио копируется (`-c:a copy`), прогресс по кадрам и ETA.
- Замер на RTX 3060 Ti: 128² — 84 мс, 256² — 297 мс, 512² — 1242 мс на тайл;
  загрузка CUDA-сессии и модели ~2,9 с переиспользуется постоянным процессом.
- Настройки: `upscaleModelPath`, `upscaleTile`, `upscaleOverlap`, `upscaleCodec`,
  `upscaleQuality`; doctor проверяет воркер, модель, ORT-библиотеку и реальную
  CUDA-сессию (`upscale-cuda`, кэш 5 минут).
- Тесты: +2 Rust (тайлы, веса) и +5 Node (валидация конфига, доступность);
  всего 34 Node + 2 Rust.

### 1.3.0 — 2026-11-23

- **`ffmpeg — operation:"raw"` стал полноценным escape hatch**: весь argv
  (`-filter_complex`, `-map`, `-c`, несколько `-i`) передаётся как есть, плюс
  `inputs[]` (сахар `-i`), `output`, `durationSec` и живой прогресс
  (`-progress pipe:1` не дублируется, `-y/-n` тоже). Для raw теперь работают
  проценты, ETA, размер и запись файла в историю.
- **Монтаж: PNG-оверлеи** — `montage.overlays[]` с `path/x/y/width/height/opacity/
  fromSec/toSec`; PNG с альфой подаётся через `-loop 1`, кладётся поверх счёта,
  окно показа через `enable='between(t,…)'`.
- **Монтаж: субтитры в том же прогоне** — `montage.subtitles` (VTT/SRT/ASS,
  `fontsDir`, `forceStyle`, `kind:"ass"`), рендер через libass после титров счёта.
- **Голос вторым входом** — `montage.audio.tracks[]` (`path/atSec/gainDb/fades`)
  микшируется `adelay→amix→loudnorm→AAC`; видео не перекодируется повторно.
- Индексы входов фиксированы: **scenes → overlays → audio**; добавлен
  `format=yuv420p` перед NVENC, doctor проверяет `overlay/subtitles/ass`.
- Проверено: raw на GPU (`scale_cuda→unsharp→cas→fps=60`, PNG-плашка, VTT,
  голос, `hevc_nvenc -gpu 1`) — 4 с/60 fps/AAC; монтаж на CPU — 8 с/60 fps,
  OCR кадра подтвердил плашку и «Счёт 1:0».

### 1.2.3 — 2026-11-23

- «Настройки» больше не пишут «Rust-воркер недоступен» и «ключа DeepSeek нет»,
  когда сервер плагина ещё старой версии и `capabilities` не отдаёт: теперь это
  нейтральное «нет данных от сервера» с подсказкой перезапустить DSH.
- Причина и путь к бинарю Rust-воркера показываются в подсказке к метке.

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

## Прогресс python-заданий

`python`-задание показывает процент, если скрипт сообщает прогресс:

- собственный протокол: печатать `progress: 42%` (старый способ);
- ffmpeg-строки: запускать ffmpeg с `-progress pipe:1 -nostats` и пробрасывать
  его stdout/stderr — парсятся `out_time_us=`, `out_time_ms=`, `time=`,
  `frame=`, `fps=`, `speed=`;
- передать `durationSec` в `mcp__gputool__python`, чтобы из `out_time` считались
  проценты и ETA.

Если скрипт запускает ffmpeg с `capture_output=True` и печатает только итог,
прогресса не будет: gputool не видит внутренний вывод. Для финальных энкодов
проще `mcp__gputool__ffmpeg` (`encode`/`raw`) — он сам добавляет
`-progress pipe:1` и даёт проценты, ETA и fps.

## Трекинг мяча (TrackNet / WASB, ONNX)

```json
{
  "source": "/video/match.mp4",
  "model": "/models/ball/tracknetv2.onnx",
  "fps": 30, "threshold": 0.5, "annotate": true,
  "outputDir": "/out/ball"
}
```

Инструмент рассчитан на ONNX-модели семейства TrackNetV2/V3 и WASB-SBDT:
`[1, frames*3, H, W]` (обычно 1×9×288×512) → heatmap `[1, 1, H, W]`. Экспорт из
PyTorch (opset 18; ORT 1.32 его понимает):

```python
torch.onnx.export(model, torch.zeros(1, 9, 288, 512), "tracknetv2.onnx",
                  opset_version=18, input_names=["input"], output_names=["output"])
```

Веса: [TrackNetV2 (PyTorch)](https://github.com/yastrebksv/TrackNet),
[TrackNetV3](https://github.com/qaz812345/TrackNetV3) (нужна ветка TrackNet),
[WASB-SBDT](https://github.com/nttcom/WASB-SBDT) (теннисные веса MIT). Модель
подключается путём `ballModelPath` или `GPUTOOL_BALL_MODEL`, в git не входит.

### Готовые веса BlurBall (настольный теннис) — рекомендуемый путь

BlurBall — форк WASB-SBDT для настольного тенниса, веса лежат прямо в
репозитории (MIT), F1 видео-инференса 0.959 по метаданным чекпойнта:

```bash
git clone --depth 1 https://github.com/mrzkp/blurball ~/.cache/gputool/blurball

/home/main/test/.venv/cv312/bin/python scripts/export_blurball_onnx.py \
  --repo ~/.cache/gputool/blurball \
  --weights ~/.cache/gputool/blurball/best_model \
  --output ~/.dsh/gputool/models/ball/blurball_tt.onnx
# вход [1,9,288,512], выход [1,3,288,512], sha256 c364b96d…
```

`best_model` — SE-вариант, `blurball_best` — базовая архитектура; скрипт
определяет вариант сам. Нормализация ImageNet включается по имени файла.

#### Альтернатива: WASB-SBDT

Веса WASB-SBDT (HRNet), MIT:

```bash
curl -L -o ~/.dsh/gputool/models/ball/wasb_tennis_best.pth.tar \
  https://raw.githubusercontent.com/ZSHYC/WASB-SBDT-FPFilter/main/pretrained_weights/wasb_tennis_best.pth.tar
# sha256: 9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb

git clone --depth 1 https://github.com/nttcom/WASB-SBDT ~/.cache/gputool/wasb-sbdt

/home/main/test/.venv/cv312/bin/python scripts/export_wasb_onnx.py \
  --repo ~/.cache/gputool/wasb-sbdt \
  --weights ~/.dsh/gputool/models/ball/wasb_tennis_best.pth.tar \
  --output ~/.dsh/gputool/models/ball/wasb_tennis.onnx
# вход [1,9,288,512], выход [1,3,288,512], sha256 8d7220ea…
```

WASB отдаёт **логиты** и обучен на ImageNet mean/std: воркер сам включает
sigmoid и нормализацию (`normalize: auto`, срабатывает по имени `wasb`), канал
берётся последний (`outputIndex`, по умолчанию `C-1`). Для других моделей
можно указать `normalize: none` и `sigmoid: false`.

Выход: `ball.jsonl` + `ball.csv` (frame/time/visible/x/y/conf), `ball.ass` и
`ball.mp4` с точкой; `annotate:false` — только координаты.

## Апскейл x2 (SPAN, только GPU)

```json
{
  "source": "/video/clip.mp4",
  "output": "/out/clip.x2.mp4",
  "batch": 8,
  "codec": "hevc", "quality": "balanced",
  "label": "x2 на GPU"
}
```

- Модель — **SPAN x2 fp16** (`span2x_ch48_fp16.onnx`, 835 КБ), инференс строго
  на CUDA: сессия создаётся с `disable_cpu_ep_fallback`, при недоступной CUDA
  задание падает (`cuda-unavailable`), CPU-режима нет.
- Кадры идут **батчами** (`batch`, 1–64, по умолчанию 8), тайлинга нет
  (`tile=0`; любой тайл — ошибка `tiling-not-supported`).
- Видео: NVDEC-декод → батч CUDA → NVENC (`hevc_nvenc`/`h264_nvenc`), аудио
  копируется. Папка PNG тоже поддерживается (декод/энкод PNG — CPU, это честно
  видно в метриках `decodeMsPerFrame`/`encodeMsPerFrame`).
- Точное разрешение: `width`/`height` (например `1280×720`) — кадр заранее
  масштабируется в `target/2`, затем x2 даёт ровно целевой размер.
- Скорость на RTX 3070, 512×288→1024×576, batch 8: **32.3 fps** (инференс
  25 мс/кадр); 2000 кадров — 65.7 с; 25 925 кадров (13 чанков) — ~14.2 мин.

Модель экспортируется из официальных весов SPAN (Apache-2.0):

```bash
# чекпойнты SPAN (архив Google Drive) -> ~/.cache/gputool/span/ckpt/spanx2_ch48.pth
git clone --depth 1 https://github.com/hongyuanyu/SPAN ~/.cache/gputool/SPAN
python scripts/export_span2x_onnx.py \
  --repo ~/.cache/gputool/SPAN \
  --weights ~/.cache/gputool/span/ckpt/spanx2_ch48.pth \
  --output ~/.dsh/gputool/models/upscale/span2x_ch48_fp16.onnx
```

Путь задаётся `upscaleModelPath` или `GPUTOOL_UPSCALE_MODEL`. Старый
4x-путь удалён: `/upscale` теперь всегда x2.


## Сырые аргументы ffmpeg

У `encode` есть `cq` (0–51): переопределяет `-cq` NVENC / `-crf` CPU-пресета.
Фильтры задаются `vf` (например `"scale=640:360,unsharp=5:5:0.5"`) или `scale`;
`args` допустим только для `raw`. Программные фильтры отключают NVDEC-декод
(энкод остаётся NVENC), чисто CUDA-цепочки (`scale_cuda`) сохраняют `-hwaccel cuda`.
`frames` учитывает `startSec`/`durationSec`: `-ss` ставится перед входом, `-t` —
после, поэтому «10 секунд с 300-й» действительно начинаются с 300-й.

Когда нужна цепочка, которой нет в готовых операциях, передаётся весь argv
(`operation: "raw"`). `inputs` подставляются как `-i` до `args`, прогресс и
`-y/-n` добавляются автоматически.

```json
{
  "operation": "raw",
  "inputs": ["/video/a.mp4"],
  "args": [
    "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
    "-i", "/video/a.mp4", "-loop", "1", "-i", "/video/plate.png", "-i", "/audio/voice.m4a",
    "-filter_complex",
    "[0:v]scale_cuda=1280:720,hwdownload,format=nv12,unsharp=5:5:0.8,cas=strength=0.5,fps=60[base];[1:v]format=rgba,scale=320:80[plate];[base][plate]overlay=24:24[s];[s]subtitles=filename='/video/fragments.ru.vtt':fontsdir='/usr/share/fonts/truetype/dejavu'[v]",
    "-map", "[v]", "-map", "2:a",
    "-c:v", "hevc_nvenc", "-gpu", "0", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "22",
    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", "4",
    "-y", "/out/final.mp4"
  ],
  "output": "/out/final.mp4",
  "durationSec": 4
}
```

Проверено на GPU: 1280×720, 60 fps, HEVC NVENC, AAC.

## Оверлеи и субтитры в монтаже

```json
{
  "operation": "montage",
  "output": "/out/final.mp4",
  "montage": {
    "scenes": [{ "input": "a.mp4", "durationSec": 8 }, { "input": "b.mp4", "durationSec": 8 }],
    "upscale": { "width": 1920, "height": 1080 },
    "fps": 60,
    "subtitles": { "path": "fragments.ru.vtt", "fontsDir": "/usr/share/fonts/truetype/dejavu" },
    "overlays": [{ "path": "plate.png", "x": 24, "y": 24, "width": 320, "height": 80, "opacity": 0.9 }],
    "score": { "events": [{ "atSec": 1, "untilSec": 12, "text": "Счёт 1 : 0" }] },
    "audio": { "tracks": [{ "path": "voice.m4a", "atSec": 2, "gainDb": -3 }] }
  }
}
```

Порядок входов: сцены → оверлеи → аудио; оверлей ложится поверх счёта,
субтитры рендерятся между ними.

### Планы

- Rust-воркеры для внутренних стадий (детекция, OCR-постобработка, трекинг) —
  см. `AGENTS.md`.
- Словарь PP-OCRv5 и профиль области табло по умолчанию.
- Прямой стрим прогресса в панель вместо опроса раз в секунду.

## Поддержка проекта

Плагин бесплатный. Если он сэкономил вам время — можно поддержать разработку.

| Сеть | Адрес |
| --- | --- |
| BTC | `bc1qmvw20wke93dtnf0hzzuqlg0fxkgxfxsnsz9xe7` |
| ETH | `0x6888d4d2Bbc1776432AA32e02f50DdBeFDeE31Af` |
| SOL | `AXAMj68npvibHCg1hNqX7b51iGTeQqKuMeaQY6asmoXL` |
| GRAM | `UQAH2_xVEAa2AkTfvZ5f5c_kLwJ7JflcEMgcKb4rWucC41q5` |
| ADA | `addr1q9lclrf04dvdfnv9n2u0nxf3ps4lwsuc0ptmuha4nk5cnxq0w5zns4f6a2qjrdnaklkgxdgmhy7tqeg99ut4enzkx84s08dqty` |
| TRX | `TCrfaBYqi4ar7n2FwrtvxT2exvbbYcToNq` |
