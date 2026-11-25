# GPU tool

Локальный MCP-сервис и панель в сайдбаре DSH для общей GPU-работы: ffmpeg
(NVDEC, NVENC), ONNX Runtime CUDA, скан лиц YuNet/SFace, OCR PP-OCRv5,
YOLO11 и произвольные GPU-скрипты Python.

GPU-процессы запускает MCP-сервер плагина, поэтому они наследуют GPU-устройства
host-процесса DSH. Bash-песочница агента не видит `/dev/dxg`; для GPU-работы
используйте плагин, а не `nvidia-smi`/ffmpeg из оболочки.

## MCP-инструменты (`mcp__gputool__*`)

| Инструмент | Назначение |
|---|---|
| `status` | GPU, сводка доктора, очередь, аренда, адреса HTTP/MCP |
| `probe` | метаданные медиафайла через ffprobe |
| `ffmpeg` | encode/decode/remux/trim/split/concat/frames/thumb/raw; NVENC, NVDEC (`hwaccel`/`cuvid`), zero-copy `scale_cuda` |
| `yolo` | YOLO11 predict или трекинг BoT-SORT на CUDA (JSONL + размеченный вывод) |
| `onnx` | универсальная сессия ONNX Runtime с провайдером CUDA (TensorRT — только если он установлен); падает вместо тихого CPU |
| `face` | скан видео YuNet + SFace с косинусным сравнением эталонного лица; `worker: auto\|rust\|python`, `batch`, `threads` |
| `ocr` | OCR по приоритету: DeepSeek Tools API, затем локальный PP-OCRv5 (`engine: auto\|deepseek\|local`, `purpose`) |
| `python` | запуск скрипта пайплайна с `CUDA_VISIBLE_DEVICES` и путями к GPU-библиотекам |
| `jobs`, `job`, `cancel` | очередь, живой прогресс (процент/ETA/fps/кадры), отмена |
| `history` | задания и вызовы: кто запустил и сколько это длилось |
| `gpu` | статус/обновление или аренда/освобождение GPU для внешнего процесса |
| `setup` | проверка возможностей; установка `torch (CUDA) + onnxruntime-gpu + ultralytics + lap` в venv плагина |
| `settings` | чтение/обновление бинарей, путей, каталогов моделей и параллельности |

Ресурсы: `gputool://doctor`, `//gpus`, `//jobs`, `//jobs/{id}`, `//history`.

Отправка задачи сразу возвращает id; прогресс опрашивается через `job`. Панель в
сайдбаре показывает те же данные с живым процентом.

## Панель в сайдбаре

Язык интерфейса: **русский по умолчанию**, английский включается переключателем
**RU / EN** в шапке панели; выбор сохраняется в `localStorage` и применяется при
следующем открытии. Строки берутся из словарей `src/client/locale.ts`, ресурсы
плагина (`locale/ru.json`, `locale/en.json`) задают заголовок и описание в
«Плагинах».

- **Обзор** — GPU/VRAM/процессы, матрица возможностей с копируемыми рецептами,
  loopback API и команда MCP.
- **Задачи** — полоса прогресса с процентом, ETA, processed/total, fps, фазой,
  прошедшим временем, тем, кто запустил, устройством, отменой и хвостом лога.
- **История** — вызовы и задания с id сессии, меткой, источником, длительностью
  и статусом.
- **Аренда** — внешние резервы GPU со сроком и освобождением.
- **Настройки** — бинари ffmpeg/ffprobe, Python, uv, каталоги моделей,
  параллельность и порт API.

## HTTP API

MCP-сервер слушает loopback; плагин проксирует его под `/gputool/api/*` для
панели. Прямое использование (те же данные, токен из
`$DSH_HOME/gputool/endpoint.json`):

```bash
curl -s http://127.0.0.1:PORT/api/state -H "Authorization: Bearer $TOKEN"
curl -s -X POST http://127.0.0.1:PORT/api/jobs \
  -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"type":"ffmpeg","operation":"encode","input":"in.mp4","output":"out.mp4","codec":"hevc"}'
```

Маршруты: `state`, `doctor`, `gpu`, `jobs`, `jobs/:id`, `jobs/:id/log`,
`jobs/:id/cancel`, `history`, `calls`, `hint`, `leases`, `lease`,
`leases/:id/release`, `setup`, `settings`.

## Установка окружения

`mcp__gputool__setup` с `install-runtime` создаёт изолированный venv в
`$DSH_HOME/gputool/venv` (uv, Python 3.12) и ставит:
`torch`/`torchvision` с настроенного CUDA-индекса (`cu126` по умолчанию; при
отсутствии колеса под версию Python uv берёт актуальное CUDA-колесо), пакеты
`nvidia-*` с cuDNN для ONNX Runtime, `onnxruntime-gpu`, `ultralytics`,
`opencv-python-headless`, `numpy` и `lap` (нужен трекеру BoT-SORT).
Существующие venv не изменяются.

Проверено на этой машине: `torch 2.15.0+cu132`, `cuda.is_available()=True`
(устройство выбирается через `CUDA_VISIBLE_DEVICES`), `onnxruntime-gpu 1.32`
с `CUDAExecutionProvider`.

Плагин никогда не запускает `sudo`/`apt`: если чего-то не хватает, доктор
показывает точную команду для ручного запуска.

## Rust-воркер лиц

`rust/face-worker` собирается в `bin/gputool-face` (`npm run build:rust`) и
работает постоянным процессом: один `onnxruntime` Environment, сессии YuNet и
SFace создаются при первом задании и пересоздаются, когда меняется каталог
моделей, размер входа или батч (в лог пишется `face: reloading engine (was …)`);
кадры читаются из ffmpeg, ресайз идёт по
предвычисленным весам, инференс — батчами до 8 кадров, NMS и выравнивание по
Umeyama — на Rust. Движок держит воркер между заданиями (`faceWorker: auto`),
сам определяет по событию `ready`, прикрепился ли CUDA EP, и при отсутствии
бинаря или падении воркера откатывается на `python/face_worker.py`.

Замер на клипе 20 с (200 кадров, 3 лица в кадре, эталон): Python — 24,2 с,
Rust тёплый — 3,14 с, Rust холодный — 4,28 с. Без эталона Rust — 11,2 мс/кадр.
Боксы совпадают с Python на 83 % при IoU ≥ 0,5.

## Трекинг мяча (TrackNet/WASB)

`rust/ball-worker` → `bin/gputool-ball`: постоянный процесс (`--serve`) и
`--config`. Модель — ONNX `[1, frames*3, H, W]` (auto-определение числа кадров
пробой 3/1 и размера из параметров, по умолчанию 512×288), output — heatmap.
CUDA EP при наличии; `device:"auto"` допускает CPU с предупреждением.
Вход нормализуется (ImageNet mean/std для WASB, иначе 0..1), выход-логиты при необходимости проходит sigmoid; канал выбирается `outputIndex`. Постобработка: argmax + центроид 5×5, порог, отбрасывание телепортов, линейное
заполнение пропусков ≤ `maxGapFrames`, медиана окном 3. Выход: `ball.jsonl`,
`ball.csv`, `ball.ass`, `ball.mp4` (subtitles + NVENC, аудио copy).

Сессия переживает задания, поэтому у неё есть подпись загрузки (модель, число
кадров, раскладка, размер входа, устройство, нормализация): при её смене сессия
пересоздаётся и в лог уходит `ball: reloading model (was …)`. Замена файла весов
по тому же пути не отслеживается — нужен перезапуск воркера.

`bgMode=concat` работает и на видео: фон — попиксельная медиана первых
`backgroundSamples` кадров диапазона (по умолчанию 32) или готовая картинка
`background`; каналы подаются как `[фон, кадры…]`, как в обучении TrackNetV3.

Раскладка входа определяется по метаданным ONNX: из формы входа берётся число
каналов `C`, из формы выхода — `L` (число heatmap). `frames=L`, а режим —
`L*3` (обычные кадры), `(L+1)*3` (concat), `L` (subtract) или `L*4`
(subtract_concat); при неоднозначности приоритет у запрошенного `bgMode`, затем
у concat. Если размеры динамические, идёт перебор `frames=1..=8`; явные
`inputWidth`/`inputHeight` переопределяют размеры из модели. Из `L` heatmap по
умолчанию берётся последняя (самый новый кадр окна), `outputIndex` переопределяет
выбор; временной ансамбль оригинала не реализован. `subtract`/`subtract_concat`
на видео по-прежнему не поддерживаются.

## GPU-апскейл x2 (SPAN fp16)

`rust/upscale-worker` → `bin/gputool-upscale`: ORT CUDA + `disable_cpu_ep_fallback`,
вход/выход строго float16, фактор проверяется как ровно x2. Батч кадров
(`batch`, 1–64), тайлинга нет. Видео: NVDEC → батч → NVENC, аудио copy; папки
PNG декодируются/энкодируются через `image` (CPU) с раздельными метриками.
Постоянная сессия пересоздаётся при смене модели, провайдера или каталога кэша
TensorRT (`upscale: reloading session (was …)`).
При OOM/ошибке CUDA задание падает (`cuda-unavailable`), CPU-фолбэка нет.
Экспорт модели — `scripts/export_span2x_onnx.py` (SPAN ch48/ch52, dynamic axes,
opset 18, fp16).


## Монтаж одним прогоном

`operation:"montage"` собирает один `-filter_complex`: сцены нормализуются
(`trim/setsar/scale/pad/fps`), склеиваются `concat`, затем апскейл (EWA через
libplacebo или lanczos), интерполяция (`fruc_vulkan` только с аппаратным Vulkan,
иначе `minterpolate`, профиль `fast` — дублирование кадров), титры счёта
(`drawtext`), субтитры (`subtitles`/`ass`), PNG-оверлеи (`overlay` с `enable`),
микс голоса (`adelay→amix→loudnorm`) и NVENC. Порядок входов:
**сцены → оверлеи → аудио**. Граф передаётся аргументом `-filter_complex`
(не все сборки знают `-filter_complex_script`).

`operation:"raw"` — полный passthrough argv с автоматическим `-progress`,
`-y/-n`, `inputs[]`-сахаром, `durationSec` и записью `output` в историю.

## Требования и ограничения

- Драйвер NVIDIA и проброс CUDA в WSL: `/dev/dxg` должен быть виден
  host-процессу DSH.
- Сборка ffmpeg с NVENC/CUVID (например `ffmpeg-nvenc`, задаётся в настройках).
- ONNX Runtime CUDA требует cuDNN 9; его приносит CUDA-колесо torch.
- Поддержки Vulkan нет: в WSL без NVIDIA-ICD устройство поднималось на llvmpipe
  (ПО), а интерполяция `fruc_vulkan` требует `VK_NV_optical_flow`, поэтому путь
  удалён. GPU-работа идёт через CUDA (NVDEC/NVENC/`scale_cuda`).
- API доступен только на loopback и защищён токеном. Задачи выполняются с
  правами пользователя хоста вне файловой песочницы агента; отправлять их
  должны только доверенные локальные вызывающие.
- Плагин не обращается к `/mnt`.
