# GPU tool

English | [Русский](README.md)

A DSH plugin and local MCP service for shared GPU work: ffmpeg NVDEC/NVENC,
ONNX Runtime CUDA (YuNet/SFace), OCR (DeepSeek → PP-OCRv5), YOLO11 and GPU scripts. Every job has
live progress (percent, ETA, fps), and the call history records which session
started it and how long it took.

- MCP tools: `mcp__gputool__{status,probe,ffmpeg,yolo,onnx,face,ocr,python,jobs,job,cancel,history,gpu,setup,settings}`
- Panel: left sidebar → GPU tool (Overview / Jobs / History / Leases / Settings)
- Interface language: **Russian by default**, English via the RU / EN switch in the panel header
- HTTP API: `/gputool/api/*` (loopback, token protected)
- Runtime setup: `mcp__gputool__setup` → `install-runtime` (torch CUDA, onnxruntime-gpu, ultralytics, lap)

Details: [docs/DETAIL.md](docs/DETAIL.md).

```bash
npm install --cache ./.npm-cache
npm run build
npm test
```

Then install the directory with the DSH plugin manager.

---

## Changelog

### 1.0.0 — 2026-11-23

First working release: the GPU MCP service, the sidebar panel, live progress and
call history.

**Added**

- **MCP server** (`src/mcp`, TypeScript): stdio transport and a loopback HTTP API
  in one process. Tools: `status`, `probe`, `ffmpeg`, `yolo`, `onnx`, `face`,
  `ocr`, `python`, `jobs`, `job`, `cancel`, `history`, `gpu`, `setup`, `settings`;
  resources `gputool://doctor|gpus|jobs|jobs/{id}|history`.
- **ffmpeg on the GPU**: NVDEC (`-hwaccel cuda`, `*_cuvid`), zero-copy
  `scale_cuda` and NVENC (`h264/hevc/av1`, `p1–p7`, `cq`, `-tag:v hvc1`) from the
  `ffmpeg-nvenc` build. Operations: **encode, decode, remux, trim, split,
  concat, frames, thumb, raw**; automatic codec choice (NVENC → CPU).
- **OCR with DeepSeek priority**: `engine="auto"` (default) calls the **DeepSeek
  Tools API** (`POST /v1/tools/ocr`, `text_recognition`) first, taking the key from
  the DSH credential store (`DEEPSEEK_API_KEY`) and never writing it to config;
  when the key or the network is unavailable it falls back to the local
  **PP-OCRv5** CUDA pipeline. `engine="local"` forces the local path and
  `engine="deepseek"` uses the API only.
- **ONNX Runtime CUDA**: generic `onnx`, **YuNet + SFace** face scan and the local
  **PP-OCRv5** OCR (detection + recognition) through `CUDAExecutionProvider`,
  failing instead of silently using CPU when `requireGpu` is set.
- **YOLO11** on CUDA: `predict` and **BoT-SORT** tracking (`track=true`,
  `tracker=botsort.yaml`), JSONL with boxes, `trackId` and object centres.
- **Python runner** for arbitrary pipelines: `CUDA_VISIBLE_DEVICES`,
  `LD_LIBRARY_PATH` assembled from the venv (torch and `nvidia-*` packages), and
  `progress: N%` lines mirrored into job progress.
- **Queue and leases**: one job per GPU and two on CPU, priority by free VRAM, a
  device lease for external processes, cancellation, and restart recovery with
  the `interrupted` code.
- **Live progress**: percent, phase, `processed/total`, fps, an ETA from the
  average speed, frame counters and a `stalled` flag.
- **History**: who (session id + label + `tool|http|mcp` source), duration,
  status, device, outputs and the log tail; filters by session, kind or operation.
- **Panel** (React + minified Vite IIFE): Overview, Jobs, History, Leases and
  Settings tabs; a capability matrix with copyable fixes; an **RU / EN** language
  switch (Russian by default, the choice persists in `localStorage`).
- **Capability doctor**: `/dev/dxg`, nvidia-smi, NVENC/CUVID, `scale_cuda`,
  ONNX Runtime and cuDNN, torch CUDA, ultralytics, and the YOLO/OCR/face models —
  each with an exact fix.
- **setup**: an isolated venv `$DSH_HOME/gputool/venv` (uv, Python 3.12) with
  CUDA torch, cuDNN, `onnxruntime-gpu`, `ultralytics`, `lap`,
  `opencv-python-headless` and `numpy`, verified with `torch.cuda.is_available()`.
- **DSH plugin** (`src/index.ts`): mounts `@deepseek-ai/dsh-mcp-client`, resolves
  it from the DSH installation, proxies `/gputool` to the loopback API and ties
  each call to its session through `tools/pre-execute`.
- **Tests**: 13 `node --test` tests covering the nvidia-smi/ffmpeg/progress
  parsers, argv planning, the stable hint hash, the MCP handshake and the client
  bundle contract.

**Fixed along the way**

- A duplicated ffmpeg binary path in argv.
- `CUDA_VISIBLE_DEVICES` for ffmpeg broke `-hwaccel_device`; the device is now
  selected only through `-gpu`/`-hwaccel_device`.
- `out_time_ms` is in microseconds — percent and ETA no longer jump.
- Vulkan support (`libplacebo` + `fruc_vulkan`) was removed: in WSL without an
  NVIDIA ICD the device ran on software llvmpipe and interpolation failed on the
  missing `VK_NV_optical_flow`. The CUDA paths (NVDEC/NVENC/`scale_cuda`) remain.
- `LD_LIBRARY_PATH` assumed `python3.12`; site-packages is discovered instead, so
  cuDNN from the venv is found.
- TensorRT in the provider list failed without its libraries — providers now
  default to CUDA → CPU.
- OCR: 3-channel input, merging of adjacent line boxes, vertical text
  orientation and correct PP-OCRv5 dictionary indexing.
- Panel: translation moved into component state (`tr`), so the language switch
  works without a reload.

**Known limitations**
- The `ppocrv5_dict.txt` shipped in `ocr-ru` is not a Cyrillic vocabulary
  (851 entries); a proper PP-OCRv5 dictionary is needed for the Russian board.
- The browser panel was not verified visually (the GUI requires a token URL).

### 1.0.1 — 2026-11-23

- Removed Vulkan support: the `libplacebo`/`fruc_vulkan` filters, the `vulkan`
  ffmpeg option, its GPU requirement, and three doctor checks (`vulkan-device`,
  `vulkan-filters`, `vulkaninfo`). Reason: the host has no GPU ICD, the chain ran
  on llvmpipe, and interpolation needs `VK_NV_optical_flow`. Interface: Russian by
  default plus English (RU/EN).
- README in Russian and English, sharing one changelog.

### 1.1.0 — 2026-11-23

- **OCR through the DeepSeek Tools API** (`POST /v1/tools/ocr`) with the priority
  DeepSeek → local PP-OCRv5: `engine: auto|deepseek|local`, `purpose:
  text_recognition|rich_layout_recognition`, page-by-page output in `ocr.jsonl`.
- `DEEPSEEK_API_KEY` is resolved from the DSH credential store and handed to the
  MCP server only through the child process environment; error text is redacted
  (`Bearer …`, `sk-…`, the key itself).
- Verified live: `SCORE 11` and `:score: 3 : 2` recognised in about 3 s with
  tokens and timing in the job log; HTTP 401/network errors fall back to local OCR.
- New settings: `ocrEngine`, `deepseekBaseUrl`, `deepseekApiKeyEnv`,
  `deepseekTimeoutMs` (170 s, under the API's 180 s limit).
- Tests: +5 (response parsing, multipart against a local server, the key never
  echoed, file limits) for 17 total.

### Planned

- Rust workers for internal stages (detection, OCR post-processing, tracking) —
  see `AGENTS.md`.
- A PP-OCRv5 dictionary and a default scoreboard region.
- Pushing progress to the panel instead of polling once per second.
