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
| `face` | скан видео YuNet + SFace с косинусным сравнением эталонного лица |
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
