# GPU tool

English | [Русский](README.md)

A DSH plugin and local MCP service for shared GPU work: ffmpeg NVDEC/NVENC,
ONNX Runtime CUDA (YuNet/SFace, TrackNetV2/V3, BlurBall, WASB, SPAN x2), OCR (DeepSeek → PP-OCRv5),
YOLO11, ball tracking and GPU scripts. Every job has live progress (percent,
ETA, fps), and the call history records which session started it and how long
it took.

- MCP tools: `mcp__gputool__{status,probe,ffmpeg,yolo,onnx,face,ocr,ball,ball_image,upscale,python,jobs,job,cancel,history,gpu,setup,settings}`
- Panel: left sidebar → GPU tool (Overview / Jobs / History / Leases / Settings)
- Interface language: **Russian by default**, English via the RU / EN switch in the panel header
- HTTP API: `/gputool/api/*` (loopback, token protected)
- **x2 upscaling (SPAN fp16)**: CUDA or TensorRT (fp16, engine cache), batching,
  NVDEC/NVENC, no tiling, no CPU fallback
- Ball tracking: **TrackNetV2/V3** (including concat background on video), **BlurBall**,
  **WASB**; the input layout is detected from the ONNX model → coordinates and a video with the dot
- On-disk models: `$DSH_HOME/gputool/models/ball/{blurball_tt,tracknetv3_badminton,wasb_tennis}.onnx`
  (ball) and `$DSH_HOME/gputool/models/upscale/span2x_ch48_fp16.onnx` (x2 upscale);
  override via `ballModelPath` / `upscaleModelPath` or the `model` parameter
- Runtime setup: `mcp__gputool__setup` → `install-runtime` (torch CUDA, onnxruntime-gpu, ultralytics, lap)

Current version: **1.7.4**. Details: [docs/DETAIL.md](docs/DETAIL.md).

## Installation

Full preparation in one command — no `sudo`/`apt`, no manual DSH profile edits:

```bash
./install.sh                 # build + GPU venv + core weights + settings
./install.sh --models all    # plus the official TrackNetV3 weights (Google Drive)
```

What `install.sh` does:

1. checks `node >= 20`, `npm`, `cargo`, `git`, `python3`, `ffmpeg`/`ffprobe`;
   a missing GPU is a warning, not a failure;
2. installs dependencies and builds: `npm install --cache ./.npm-cache`,
   `npm run build`, `npm run build:rust` (workers in `bin/`);
3. creates `$DSH_HOME/gputool/venv` (via `uv` when available, otherwise
   `python3 -m venv`) and installs torch cu126, onnxruntime-gpu, ultralytics,
   opencv-python-headless, numpy, lap;
4. prepares weights in `$DSH_HOME/gputool/models`: **BlurBall**, **WASB-SBDT**
   and **SPAN x2** (ch48 + ch52); `--models all` also adds **TrackNetV3**
   (`TrackNetV3_ckpts.zip` from Google Drive via `gdown`). Repository clones and
   archives go to `$DSH_HOME/gputool/cache`; existing files are not downloaded
   again;
5. writes `manifest.json` (sha256 of every model) and updates `settings.json`:
   `pythonPath`, `ballModelPath`, `upscaleModelPath`, GPU
   `ffmpegGpuPath`/`ffprobeGpuPath` (when an NVENC build is found);
6. reminds that the plugin itself is installed through the DSH plugin manager
   (`install_bundle` with the directory path): the profile is not edited by hand.

| Flag | Effect |
| --- | --- |
| `--models core\|all\|none` | which weights to prepare (default `core`) |
| `--skip-models` | leave the weights alone |
| `--force-models` | rebuild/re-download the weights |
| `--skip-build` | skip Node/Rust builds |
| `--skip-runtime` | skip GPU venv creation |
| `--verify-models` | verify models against `manifest.json` |
| `--no-manifest` | do not rewrite `manifest.json` |
| `--check` | check the environment and print the plan |
| `--dry-run` | print commands without running them |

After installation: install the directory with the DSH plugin manager, restart
DSH and open the GPU tool panel (Settings → Check). `venv/`, `cache/` and
`models/` are not tracked by git.

### Manual build

```bash
npm install --cache ./.npm-cache
npm run build        # TypeScript + client bundle
npm run build:rust   # face / upscale / ball workers (cargo)
npm test             # Node tests
npm run test:rust    # worker tests
```

---

## Admin API

- `POST /api/admin/reap` — fail stalled jobs (dead pid/no progress) and unblock
  the queue.
- `POST /api/admin/restart` — restart the MCP service process (DSH brings it
  back). Running jobs are marked `interrupted` on boot.

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

### 1.7.4 — 2026-11-25

- **Automatic ONNX layout detection**: `frames`/`bgMode` are no longer required.
  The worker reads the input/output shapes (`[1,C,288,512]`, `[1,L,…]`) and
  derives `frames=L` and the layout: `L*3` — plain frames, `(L+1)*3` — concat,
  `L` — subtract, `L*4` — subtract_concat. Ambiguity is resolved by the
  requested `bgMode`, then by `concat`. `tracknetv3_badminton.onnx` (27
  channels, 8 heatmaps) now loads without `frames` — that was exactly why 1.7.3
  failed (`model did not accept 512x288 with 3 or 9 channels`).
- **Diagnostics**: a failure now reports the input/output shape and dtype, the
  list of tried `(frames, channels, layout)` candidates and the real ORT error
  (e.g. `Got: 28 Expected: 27`) instead of "3 or 9 channels".
- **Heatmap selection**: the last heatmap of the window (newest frame) is used
  by default; `outputIndex` overrides it, and `heatmaps` is returned in the
  result.
- Tests: Rust ball 14, Node 62.

### 1.7.3 — 2026-11-25

- **TrackNetV3 on video**: `bgMode=concat` now works for videos, not only still
  images. The background is the per-pixel median of the first frames of the
  range (like `np.median` in the official TrackNetV3) or an explicit
  `background` image; channels are laid out as `[background, frames…]` exactly
  as in training. New `background` and `backgroundSamples` (8–128, default 32)
  parameters on the `ball` MCP tool.
- `subtract`/`subtract_concat` remain still-image only.
- The background actually used is logged and returned
  (`background=median of N frames`).
- Tests: Rust ball 12, Node 62.

### 1.7.2 — 2026-11-25

- **Model switching in a persistent worker**: `ball`, `upscale` and `face` loaded
  their session once per process, so a second `ball` job with a different model
  silently ran on the first one (BlurBall and TrackNetV3 runs on the same clip
  produced byte-identical `ball.jsonl`/`ball.csv`). The session is now recreated
  whenever the load signature changes: model path, frames/layout/size/device/
  normalization (ball), model/provider/TensorRT cache (upscale), model
  directory/input size/batch (face); the log line is `reloading model (was …)`.
- **`ball_image`**: coordinates were dropped while parsing the worker result in
  Node — `ball_image.json` contained only `model` and no annotated PNG was
  produced. `x/y/confidence/visible/annotated` now reach the job.
- Model weights unchanged: manifest 11/11 matches by sha256.
- Tests: ball 9, upscale 5, face 4; Node 62.

### 1.7.1 — 2026-11-25

- **Stuck at 100%**: the `upscale` and `ball` workers stopped reading frames at
  the computed `total_frames`, while `-t` and the fps filter emit a few frames
  more. The decoder stayed blocked on a full pipe and `wait()` hung forever, so
  the job froze at 100%. Reading now continues to EOF; `total_frames` only
  drives percent and ETA.
- **ffmpeg stderr** is drained by a separate thread, so a full 64 KB pipe can no
  longer block the decoder or encoder.
- Panel: phases are localized (`frames`, `finalizing the file`, `ball tracking`,
  `faces`); after 100% the job shows finalization instead of frozen frames.
- Tests: 60 Node, Rust `ball`/`upscale`, `clippy -D warnings`.

### 1.7.0 — 2026-11-25

- **`ball_image`**: still-image ball capture (or 1–8 images) in Rust: heatmap →
  subpixel coordinates, `ball_image.json` and a PNG with a red marker. Supports
  `bgMode=concat` for official TrackNetV3 checkpoints.
- **TrackNetV3 weights**: official `TrackNet_best.pt`/`InpaintNet_best.pt`
  downloaded (Google Drive `TrackNetV3_ckpts.zip`) and exported to
  `tracknetv3_badminton.onnx` (27×288×512 → 8 heatmaps). On a table-tennis
  frame the official (badminton) weights scored conf 0.027 — no ball; BlurBall
  scored 0.255 on the same frame. Proper TT weights are still missing (OneDrive
  access denied).
- **SPAN + TensorRT**: TensorRT 10.16.1 (cu13) installed in the GPU venv; the
  worker now accepts `provider=auto|cuda|tensorrt` with fp16 and an engine cache.
  16-frame benchmark: CUDA 25.8 fps (32.6 ms/frame), TensorRT with a warm cache
  43.9 fps (16.4 ms/frame); first engine build ≈54 s, cache in
  `models/upscale/trt-cache`.
- `manifest.json` with all weights (12 files, sha256).
- Tests: 60 Node (+1), Rust ball/upscale.

### 1.6.3 — 2026-11-24

- **Python job progress**: besides `progress: NN%`, ffmpeg lines
  `out_time_us=`/`out_time_ms=`/`time=`, `frame=`, `fps=`, `speed=` are parsed;
  the `durationSec` parameter enables percent and ETA.
- An empty percent no longer looks broken: the panel shows live elapsed time and
  a “the script reports no progress” hint.
- Why `final2.py` had no progress: the script runs ffmpeg with
  `capture_output=True` and prints only `START`, so gputool never sees
  `-progress`. Fix: run the encode via `ffmpeg raw` (it injects
  `-progress pipe:1`) or forward ffmpeg lines from the script.
- Tests: 59 Node (+5 for the progress parser).

### 1.6.2 — 2026-11-24

- **`vf` is no longer dropped**: the `vf` parameter (full filter chain) works in
  `encode`, `trim`, `split`, `frames`, `thumb`; `scale` is joined with it. Root
  cause: `args` was honoured only by `raw`, while `encode` silently ignored
  `-vf` — now `args` outside `raw` fails with `args-raw-only`.
- **GPU filters**: software filters (`scale`, `unsharp`, `crop`…) automatically
  disable NVDEC decode (warning in job.warnings) while encoding stays NVENC.
  Pure CUDA chains (`scale_cuda`, `libplacebo`) keep `-hwaccel cuda`.
- **`encode`** now honours `startSec`/`durationSec` (`-ss` before input, `-t`
  after) and `codec:"copy"` with filters fails with `filters-with-copy` instead
  of silently dropping them.
- **`frames`** applies `scale`/`vf` together with `fps` (`fps=1,scale=640:360`).
- **`/api/state.ffmpeg`**: binary, NVENC/CPU encoders and filters availability
  (`scale_cuda`, `hwdownload`, `unsharp`, `cas`, `subtitles`, `libplacebo`),
  hwaccels; `job.warnings` is exposed in `/api/jobs` and the panel.
- Verified on GPU: a 3 s 1280×720 clip with `vf:"scale=640:360"` produced
  640×360, h264_nvenc, duration 3.000 s.
- Tests: 54 Node (+8: vf/copy/seek/frames/diagnostics).

### 1.6.1 — 2026-11-24

- **Queue watchdog**: child pids are tracked (`ctx.trackPid`); a dead process
  becomes `failed/process-died`, missing progress → `stalled`, timeout →
  `timeout`; the slot is released and the queue continues.
- **Running cancel is hard now**: SIGTERM→SIGKILL by pid, and after 5 s the job is
  force-finalized (`cancelled`) even if the executor hangs.
- **Admin API**: `POST /api/admin/reap` and `POST /api/admin/restart`; the panel
  gets a “Reap stalled” button and a “stalled” badge.
- **State**: `/api/state` and `/api/jobs` expose `alive`, `stalled`, `pids`;
  `queue` gains `stalled`/`aliveRunning`.
- **ffmpeg encode**: `cq` (0–51) overrides the preset `-cq`/`-crf`.
- **ffmpeg frames**: honours `startSec`/`durationSec` (`-ss` before input, `-t`
  after) — “10 seconds from 300” no longer starts at the beginning.
- **upscale**: `width`/`height` give an exact target resolution (pre-resize to
  target/2, then x2), e.g. 1280×720 → 1280×720 without a second pass.
- **GPU**: `usedByProcessesMB`/`otherMB` show VRAM used outside the namespace
  (another container/host); auto-selection logs the chosen GPU and free MB and
  avoids cards with <1 GB free.
- **device**: a numeric string (`"1"`) is accepted as an index; garbage gets
  `device must be "auto", "cpu" or an integer GPU index`.

### 1.6.0 — 2026-11-24

- **Upscaling is x2-only now** (4x-UltraSharp removed): lightweight **SPAN x2
  fp16** instead of RRDBNet, inference strictly on CUDA with no CPU fallback.
- **fp16 + batching**: frames run in batches (`batch`, default 8), one
  `session.run` per batch; tiling removed (`tile=0`), any tile value is an error.
- **All weights downloaded**: SPAN archive (ch48/ch52) and Real-ESRGAN x2plus;
  export via `scripts/export_span2x_onnx.py` → `span2x_ch48_fp16.onnx` (835 KB).
- **fp16 quality**: PSNR vs PyTorch fp32 — **57.99 dB**, MAE 0.26 levels.
- **Speed** on RTX 3070 (512×288→1024×576, batch 8): **32.3 fps**; inference
  24.1 ms/frame, PNG decode 1.8 ms, PNG encode 5.0 ms.
- **Video path**: NVDEC → CUDA batch → NVENC, 3 s 512×288→1024×576 clip in 2.8 s
  (~27 fps), HEVC.
- **Chunk c00**: 2000 frames in **65.7 s**; ETA for all 13 parts (25,925 frames)
  — **~13.4 min** (old 4x: 231 ms/frame ≈ 4.3 fps ≈ 100 min).
- Settings: `upscaleModelPath` (default `span2x_ch48_fp16.onnx`), `upscaleBatch`;
  `upscaleTile`/`upscaleOverlap` removed; doctor checks the fp16 session and x2.

### 1.5.2 — 2026-11-24

- **BlurBall is now the primary ball detector** (table tennis, MIT): weights live
  in `mrzkp/blurball` (`best_model`, SE variant, checkpoint video-inference F1
  **0.959**), exported to ONNX by `scripts/export_blurball_onnx.py` (auto SE
  detection, input `[1,9,288,512]`, output `[1,3,288,512]`).
- ImageNet normalization is now auto-enabled for `blurball` filenames too; no
  extra sigmoid is applied because BlurBall already outputs probabilities.
- Real 1280×720@25 clip (20 s): **488/502 frames** with the ball (324 real
  detections vs 274 for WASB), mean confidence **0.794**, 14.5 s including
  annotation; the dot was verified pixel-wise (conf 0.965).
- README recommends BlurBall, WASB stays as an alternative; version 1.5.2.

### 1.5.1 — 2026-11-24

- **Real WASB-SBDT weights downloaded** (`wasb_tennis_best.pth.tar`, 6.1 MB, MIT)
  and exported to ONNX by `scripts/export_wasb_onnx.py` (HRNet, input
  `[1,9,288,512]`, output `[1,3,288,512]`, opset 18).
- The `ball` worker now selects a channel from multiple heatmaps (`outputIndex`,
  defaults to the last), applies **sigmoid** for logits and **ImageNet
  normalization** for WASB models (`normalize: auto|imagenet|none`), and rejects
  teleports (`maxJumpPx`, 30 % of frame width by default).
- Verified on a real 1280×720@25 clip (20 s, WASB): **456/502 frames** with the
  ball (274 real detections, mean confidence 0.78), median step 9.4 px, ~13.7 s
  total including annotation, HEVC `ball.mp4`.
- The dot was verified pixel-wise: at a real detection (conf 0.94) 29 yellow
  pixels sit exactly at the model coordinates, while the original frame has 0.
- README documents download/export, sha256, normalization and the new parameters.

### 1.5.0 — 2026-11-24

- **Table tennis ball tracking** (`mcp__gputool__ball`): ONNX TrackNetV2/V3 and
  WASB models (3 frames → heatmap) through the new Rust worker
  `bin/gputool-ball`; CUDA when available, otherwise an honest CPU fallback for
  `auto`.
- Pipeline: ffmpeg → frame ring buffer → bilinear resize 512×288 → tensor
  `[1, frames*3, H, W]` → heatmap → subpixel center (argmax + centroid), jump
  rejection, gap filling up to `maxGapFrames`, median smoothing.
- Outputs: `ball.jsonl` (frame/time/visible/x/y/conf), `ball.csv`, `ball.ass` and
  a video with the dot burned in (`subtitles` filter + NVENC, audio copied).
- Settings: `ballModelPath`, `ballThreshold`, `ballMaxGapFrames`, `ballAnnotate`,
  `ballDevice`; panel fields and doctor checks (`ball-worker`, `ball-model`).
- The model is not committed: README documents the ONNX export (`opset 18`,
  `[1,9,288,512]`) for TrackNetV2/V3 and WASB.
- Tests: +3 Rust (heatmap/postprocess/ASS) and +3 Node (validation) — 37 Node,
  8 Rust total.

### 1.4.0 — 2026-11-24

- **GPU 4x-UltraSharp upscaling** (`mcp__gputool__upscale`): a separate Rust worker
  `bin/gputool-upscale` that runs inference **strictly on CUDA** — the session is
  created with `session.disable_cpu_ep_fallback`, so the model is either fully on
  the GPU or the job fails with `cuda-ep-unavailable`/`gpu-unsupported` (no CPU
  fallback).
- **Images**: tiled with overlap (256/16 by default), linear feathered blending
  without seams, RGBA keeps transparency (alpha via Lanczos).
- **Video**: NVDEC decode → frames → CUDA inference → NVENC
  (`hevc_nvenc`/`h264_nvenc`), audio is copied (`-c:a copy`), per-frame progress
  and ETA.
- Measured on RTX 3060 Ti: 128² tile — 84 ms, 256² — 297 ms, 512² — 1242 ms;
  the ~2.9 s CUDA session/model load is reused by the persistent process.
- Settings: `upscaleModelPath`, `upscaleTile`, `upscaleOverlap`, `upscaleCodec`,
  `upscaleQuality`; doctor checks the worker, model, ORT library and a real CUDA
  session (`upscale-cuda`, 5-minute cache).
- Tests: +2 Rust (tiles, weights) and +5 Node (config validation, availability);
  34 Node + 2 Rust total.

### 1.3.0 — 2026-11-23

- **`ffmpeg — operation:"raw"` is now a real escape hatch**: the whole argv
  (`-filter_complex`, `-map`, `-c`, multiple `-i`) passes through untouched, plus
  `inputs[]` sugar, `output`, `durationSec` and live progress (`-progress pipe:1`
  and `-y/-n` are never duplicated). Raw jobs now report percent, ETA, size and the
  artifact in history.
- **Montage PNG overlays** — `montage.overlays[]` with `path/x/y/width/height/
  opacity/fromSec/toSec`; alpha PNG is looped with `-loop 1`, composited above the
  score, visibility window via `enable='between(t,…)'`.
- **Subtitles in the same run** — `montage.subtitles` (VTT/SRT/ASS, `fontsDir`,
  `forceStyle`, `kind:"ass"`) through libass after the score drawtext.
- **Voice as a second input** — `montage.audio.tracks[]` (`path/atSec/gainDb/fades`)
  mixed with `adelay→amix→loudnorm→AAC`; video is not re-encoded twice.
- Input order is fixed: **scenes → overlays → audio**; `format=yuv420p` added before
  NVENC; doctor checks `overlay/subtitles/ass`.
- Verified: raw on GPU (`scale_cuda→unsharp→cas→fps=60`, PNG plate, VTT, voice,
  `hevc_nvenc -gpu 1`) — 4 s/60 fps/AAC; CPU montage — 8 s/60 fps, frame OCR
  confirmed the plate and “Счёт 1:0”.

### 1.2.3 — 2026-11-23

- Settings no longer claim “Rust worker is unavailable” or “no DeepSeek key” when
  the plugin server is still an older build without `capabilities`: it now shows a
  neutral “no data from the server” with a hint to restart DSH.
- The Rust worker reason and binary path appear in the badge tooltip.

### 1.2.2 — 2026-11-23

- The **Jobs** tab now lists only current work: `running` and `queued` jobs;
  finished ones stay in History.
- The **History** tab shows the **last 10 entries** (after search and filtering)
  with a caption; clicking a row still opens the details.

### 1.2.1 — 2026-11-23

- **Fixed: the Jobs tab crashed** — the progress component did not receive the
  translator (`tr`), so any job list broke rendering.
- **Settings show the engines**: which OCR path is selected (`auto`/`deepseek`/
  `local`), whether the DeepSeek key is found and which variable holds it, where
  the local PP-OCRv5 models live, the scoreboard region, the face worker mode and
  whether the Rust worker is available.
- `/api/state` now exposes a `capabilities` block (OCR + face) so the panel reports
  the actual state instead of guessing.

### 1.2.0 — 2026-11-23

- **Rust face worker** (`rust/face-worker`, binary `bin/gputool-face`): a persistent
  process with a single `onnxruntime` environment. YuNet and SFace sessions are
  created once, frames are read from ffmpeg, prepared with precomputed resize
  weights and inferred in batches (up to 8), with NMS and alignment in Rust.
- **Measured on one clip** (20 s, 200 frames, 3 faces per frame, with reference):
  the Python worker took **24.2 s** (121 ms/frame), warm Rust **3.14 s**
  (15.7 ms/frame), cold Rust 4.28 s — about **7.7×**; without a reference Rust
  runs at 11.2 ms/frame.
- **Why**: Python rebuilds ORT sessions on every job, runs
  `estimateAffinePartial2D`/`warpAffine` per face and fights the GIL; Rust keeps
  sessions alive, and the weight-table resize cut 18.5 → 11.2 ms/frame.
- **Honesty**: the worker reports the device it actually attached (CUDA or CPU)
  in `ready` and in the result; the engine falls back to the Python path when the
  Rust binary is missing or fails.
- **Settings**: `faceWorker: auto|rust|python` (default `auto`), MCP options
  `worker`, `batch`, `threads`; build with `npm run build:rust`, test with
  `npm run test:rust`.
- **Parity**: 83 % of boxes match Python at IoU ≥ 0.5 (median 0.893); the
  difference is bilinear interpolation rounding, the same faces are found.
- Tests: +2 Node (binary availability, `libonnxruntime.so` lookup) and 3 Rust
  (`nms`, `similarity`, `cosine`).

## Python job progress

A `python` job shows a percent only when the script reports progress:

- own protocol: print `progress: 42%` (legacy);
- ffmpeg lines: run ffmpeg with `-progress pipe:1 -nostats` and forward its
  stdout/stderr — `out_time_us=`, `out_time_ms=`, `time=`, `frame=`, `fps=`,
  `speed=` are parsed;
- pass `durationSec` to `mcp__gputool__python` so `out_time` yields percent and
  ETA.

If the script runs ffmpeg with `capture_output=True` and prints only the final
result, there is no progress: gputool never sees the inner output. For final
encodes prefer `mcp__gputool__ffmpeg` (`encode`/`raw`) — it injects
`-progress pipe:1` and provides percent, ETA and fps.

## Ball tracking (TrackNetV2/V3, BlurBall, WASB, ONNX)

```json
{
  "source": "/video/match.mp4",
  "model": "/models/ball/tracknetv3_badminton.onnx",
  "bgMode": "concat",
  "backgroundSamples": 32,
  "threshold": 0.5, "annotate": true,
  "outputDir": "/out/ball"
}
```

The worker derives the input layout from the ONNX model itself: the input gives
the channel count `C` and `H×W`, the output gives `L` (number of heatmaps). Then
`frames = L` and the mode follows from `C`:

| Channels `C` | Mode | Input |
| --- | --- | --- |
| `L*3` | plain frames | `[frames…]` |
| `(L+1)*3` | `concat` | `[background, frames…]` |
| `L` | `subtract` | per-frame difference to the background |
| `L*4` | `subtract_concat` | frame RGB + difference |

So `frames` and `bgMode` may be omitted: on ambiguity the requested `bgMode`
wins, then `concat`. Explicit `inputWidth`/`inputHeight` override the model
sizes; if the shape is dynamic the worker probes `frames = 1..=8`. TrackNetV3
(`[1,27,288,512]` → `[1,8,288,512]`) runs as is, without `frames=8`.

### Background for `bgMode=concat` on video

A background is needed when the channel count is `(frames+1)*3`. The worker
takes the first `backgroundSamples` frames of the job range (default 32) and
computes a **per-pixel median** — the same idea as `np.median` in the official
TrackNetV3. Alternatively pass a ready image:

```json
{ "bgMode": "concat", "background": "/video/empty_court.png" }
```

Those frames then take part in inference (a window of `frames` frames; the
last heatmap corresponds to the current frame). `subtract`/`subtract_concat` are not
supported on video yet — still images only.

### Heatmap selection

TrackNetV3 returns `L` heatmaps per window. The last one (the newest frame) is
used by default and `outputIndex` overrides it; the original repository's
temporal ensemble is not implemented (a deliberate simplification). The number
of heatmaps is returned as `heatmaps`.

When no layout fits, the error reports the input/output shapes and dtype, the
list of tried `(frames, channels, layout)` candidates and the real ONNX Runtime
error:

```text
model-layout-unsupported: input [1, 27, 288, 512] Float32, output [1, 8, 288, 512];
tried: [(3,9,""), (8,27,"concat"), …]; last ORT error: Got: 28 Expected: 27
```

### Attaching a model

The ONNX file comes from `ballModelPath`, `GPUTOOL_BALL_MODEL` or the `model`
parameter; weights are not stored in git. Generic export (opset 18, ORT 1.32):

```python
torch.onnx.export(model, torch.zeros(1, 9, 288, 512), "tracknetv2.onnx",
                  opset_version=18, input_names=["input"], output_names=["output"])
```

Export scripts for every supported family live in `scripts/`:
[TrackNetV2](https://github.com/yastrebksv/TrackNet),
[TrackNetV3](https://github.com/qaz812345/TrackNetV3),
[WASB-SBDT](https://github.com/nttcom/WASB-SBDT), BlurBall.

#### TrackNetV3 (official weights)

```bash
git clone --depth 1 https://github.com/qaz812345/TrackNetV3 ~/.cache/gputool/tracknetv3

python scripts/export_tracknetv3_onnx.py \
  --repo ~/.cache/gputool/tracknetv3 \
  --weights ~/.dsh/gputool/models/ball/ckpts/TrackNet_best.pt \
  --output ~/.dsh/gputool/models/ball/tracknetv3_badminton.onnx
# input [1,27,288,512] (background + 8 frames), output [1,8,288,512]
```

The official weights are trained on badminton: on table tennis they find the
ball only partially (68/250 frames over a 10 s ROI versus 112/250 for BlurBall),
so BlurBall or proper TT weights remain the recommendation for table tennis
(`wasn-lab/TrackNetV3_TableTennis`, weights live behind a closed OneDrive).

#### BlurBall (table tennis) — recommended

BlurBall is a WASB-SBDT fork for table tennis; weights are in the repository
(MIT), checkpoint video-inference F1 is 0.959:

```bash
git clone --depth 1 https://github.com/mrzkp/blurball ~/.cache/gputool/blurball
python scripts/export_blurball_onnx.py --repo ~/.cache/gputool/blurball \
  --weights ~/.cache/gputool/blurball/best_model \
  --output ~/.dsh/gputool/models/ball/blurball_tt.onnx
```

`best_model` is the SE variant, `blurball_best` the baseline; the script picks
the right one. ImageNet normalization is enabled from the filename.

#### Alternative: WASB-SBDT

```bash
curl -L -o ~/.dsh/gputool/models/ball/wasb_tennis_best.pth.tar \
  https://raw.githubusercontent.com/ZSHYC/WASB-SBDT-FPFilter/main/pretrained_weights/wasb_tennis_best.pth.tar
# sha256: 9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb
git clone --depth 1 https://github.com/nttcom/WASB-SBDT ~/.cache/gputool/wasb-sbdt
python scripts/export_wasb_onnx.py --repo ~/.cache/gputool/wasb-sbdt \
  --weights ~/.dsh/gputool/models/ball/wasb_tennis_best.pth.tar \
  --output ~/.dsh/gputool/models/ball/wasb_tennis.onnx
```

WASB emits **logits** and expects ImageNet normalization; the worker enables
both automatically (`normalize: auto` matches the `wasb` filename) and takes the
last output channel (`outputIndex`). Other models can use `normalize: none` and
`sigmoid: false`.

The outputs are `ball.jsonl` + `ball.csv` (frame/time/visible/x/y/conf), `ball.ass`
and `ball.mp4` with the dot; `annotate:false` keeps coordinates only. The result
also reports `heatmaps` (the model's output channel count).

For a single image (or 1–8 frames) use `mcp__gputool__ball_image`: the same
worker runs on still images and writes `ball_image.json` plus a PNG with the
marker.

## x2 upscale (SPAN, CUDA/TensorRT)

```json
{
  "source": "/video/clip.mp4",
  "output": "/out/clip.x2.mp4",
  "batch": 8,
  "codec": "hevc", "quality": "balanced"
}
```

- Model — **SPAN x2 fp16** (`span2x_ch48_fp16.onnx`, 835 KB). Providers:
  `provider = auto | cuda | tensorrt` (auto tries TensorRT, then CUDA); there is
  no CPU mode, so without them the job fails (`cuda-unavailable` /
  `tensorrt-unavailable`). TensorRT keeps its engine cache in
  `models/upscale/trt-cache` (first build ≈54 s).
- Frames are **batched** (`batch`, 1–64, default 8); no tiling (`tile=0`, any
  tile value fails with `tiling-not-supported`).
- Video: NVDEC → CUDA batch → NVENC (`hevc_nvenc`/`h264_nvenc`), audio copied.
  PNG directories work too (PNG decode/encode is CPU, reported separately).
- Exact target resolution: `width`/`height` (e.g. `1280×720`) — the frame is
  pre-scaled to `target/2`, then x2 produces exactly the requested size.
- Speed on RTX 3070, 16 frames at 512×288, batch 8: CUDA **25.8 fps**
  (32.6 ms/frame), TensorRT with a warm cache **43.9 fps** (16.4 ms/frame).

Export from the official SPAN weights (Apache-2.0):

```bash
git clone --depth 1 https://github.com/hongyuanyu/SPAN ~/.cache/gputool/SPAN
python scripts/export_span2x_onnx.py --repo ~/.cache/gputool/SPAN \
  --weights ~/.cache/gputool/span/ckpt/spanx2_ch48.pth \
  --output ~/.dsh/gputool/models/upscale/span2x_ch48_fp16.onnx
```

Set `upscaleModelPath` or `GPUTOOL_UPSCALE_MODEL`. The old 4x path is gone.


## Raw ffmpeg arguments

For chains that the preset operations do not cover, pass the whole argv with
`operation: "raw"`. `inputs` become `-i` before `args`; progress and `-y/-n` are
added automatically.

```json
{
  "operation": "raw",
  "inputs": ["/video/a.mp4"],
  "args": ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
    "-i", "/video/a.mp4", "-loop", "1", "-i", "/video/plate.png", "-i", "/audio/voice.m4a",
    "-filter_complex", "[0:v]scale_cuda=1280:720,hwdownload,format=nv12,unsharp=5:5:0.8,cas=strength=0.5,fps=60[base];[1:v]format=rgba,scale=320:80[plate];[base][plate]overlay=24:24[s];[s]subtitles=filename='/video/fragments.ru.vtt':fontsdir='/usr/share/fonts/truetype/dejavu'[v]",
    "-map", "[v]", "-map", "2:a", "-c:v", "hevc_nvenc", "-gpu", "0", "-c:a", "aac",
    "-movflags", "+faststart", "-t", "4", "-y", "/out/final.mp4"],
  "output": "/out/final.mp4", "durationSec": 4
}
```

## Overlays and subtitles in montage

`montage.subtitles` renders VTT/SRT/ASS via libass; `montage.overlays` composites
alpha PNGs (with `x/y/width/height/opacity/fromSec/toSec`) above the score. Input
order is scenes → overlays → audio, so voice tracks stay independent.

### Planned

- TrackNetV3 TT weights (`wasn-lab/TrackNetV3_TableTennis`, closed OneDrive) for
  accurate tracking: the official badminton weights detect only partially.
- `subtract`/`subtract_concat` on video — if TT weights with that layout appear.
- The original TrackNetV3 temporal heatmap ensemble (optional).
- Rust workers for internal stages (detection, OCR post-processing, tracking) —
  see `AGENTS.md`.
- A PP-OCRv5 dictionary and a default scoreboard region.
- Pushing progress to the panel instead of polling once per second.

## Supporting the project

The plugin is free. If it saved you time, you can support development.

| Network | Address |
| --- | --- |
| BTC | `bc1qmvw20wke93dtnf0hzzuqlg0fxkgxfxsnsz9xe7` |
| ETH | `0x6888d4d2Bbc1776432AA32e02f50DdBeFDeE31Af` |
| SOL | `AXAMj68npvibHCg1hNqX7b51iGTeQqKuMeaQY6asmoXL` |
| GRAM | `UQAH2_xVEAa2AkTfvZ5f5c_kLwJ7JflcEMgcKb4rWucC41q5` |
| ADA | `addr1q9lclrf04dvdfnv9n2u0nxf3ps4lwsuc0ptmuha4nk5cnxq0w5zns4f6a2qjrdnaklkgxdgmhy7tqeg99ut4enzkx84s08dqty` |
| TRX | `TCrfaBYqi4ar7n2FwrtvxT2exvbbYcToNq` |
