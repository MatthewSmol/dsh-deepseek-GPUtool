#!/usr/bin/env bash
# Полная подготовка GPU tool: сборка, GPU-venv, веса ONNX и настройки.
#
# Скрипт не ставит системные пакеты (sudo/apt не используются) и не правит
# профиль DSH: регистрация плагина — через менеджер плагинов DSH (см. README).
# Всё скачанное складывается в $DSH_HOME/gputool/{venv,models,cache}.
#
# Примеры:
#   ./install.sh                 # сборка + venv + базовые веса + настройки
#   ./install.sh --models all    # плюс официальные веса TrackNetV3
#   ./install.sh --check         # проверить окружение и показать план
#   ./install.sh --dry-run       # напечатать команды, ничего не менять
#   ./install.sh --verify-models # сверить модели с manifest.json
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DSH_HOME="${DSH_HOME:-$HOME/.dsh}"
GPUTOOL_HOME="${GPUTOOL_HOME:-$DSH_HOME/gputool}"
VENV="${GPUTOOL_VENV:-$GPUTOOL_HOME/venv}"
MODELS="${GPUTOOL_MODELS:-$GPUTOOL_HOME/models}"
CACHE="${GPUTOOL_CACHE:-$GPUTOOL_HOME/cache}"
SETTINGS="$GPUTOOL_HOME/settings.json"
PY="$VENV/bin/python"
TORCH_INDEX="${GPUTOOL_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
PYTHON_VERSION="${GPUTOOL_PYTHON:-3.12}"

MODELS_MODE="core"     # core | all | none
DO_BUILD=1
DO_RUNTIME=1
DO_MANIFEST=1
DRY_RUN=0
CHECK=0
VERIFY=0
FORCE_MODELS=0

BLURBALL_REPO="https://github.com/mrzkp/blurball"
WASB_WEIGHTS_URL="https://raw.githubusercontent.com/ZSHYC/WASB-SBDT-FPFilter/main/pretrained_weights/wasb_tennis_best.pth.tar"
WASB_REPO="https://github.com/nttcom/WASB-SBDT"
SPAN_REPO="https://github.com/hongyuanyu/SPAN"
SPAN_CKPTS_ID="1iYUA2TzKuxI0vzmA-UXr_nB43XgPOXUg"
TRACKNETV3_REPO="https://github.com/qaz812345/TrackNetV3"
TRACKNETV3_CKPTS_ID="1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33m[предупреждение]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[ошибка]\033[0m %s\n' "$*" >&2; exit 1; }

run() {
  if (( DRY_RUN )); then
    printf '    [dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

usage() {
  sed -n '2,20p' "$ROOT/install.sh" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Флаги:
  --models core|all|none   какие веса готовить (по умолчанию core)
  --skip-models            то же, что --models none
  --force-models           пересобрать/перекачать веса заново
  --skip-build             не собирать Node/Rust
  --skip-runtime           не создавать GPU-venv и не ставить пакеты
  --verify-models          только сверить модели с manifest.json
  --no-manifest            не перезаписывать manifest.json
  --check                  только проверка окружения и план
  --dry-run                показать команды без выполнения
  -h, --help               эта справка
EOF
}

while (($#)); do
  case "$1" in
    --models) MODELS_MODE="${2:?--models требует значение}"; shift 2 ;;
    --skip-models) MODELS_MODE="none"; shift ;;
    --force-models) FORCE_MODELS=1; shift ;;
    --skip-build) DO_BUILD=0; shift ;;
    --skip-runtime) DO_RUNTIME=0; shift ;;
    --verify-models) VERIFY=1; shift ;;
    --no-manifest) DO_MANIFEST=0; shift ;;
    --check) CHECK=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "неизвестный аргумент: $1 (см. --help)" ;;
  esac
done

case "$MODELS_MODE" in
  core|all|none) ;;
  *) die "--models: ожидается core, all или none (получено: $MODELS_MODE)" ;;
esac

# ---------------------------------------------------------------- проверки

need() { command -v "$1" >/dev/null 2>&1 || die "не найден $1; установите его и повторите"; }

preflight() {
  need git
  need node
  need npm
  need python3
  need curl
  need ffmpeg
  need ffprobe
  if (( DO_BUILD )); then need cargo; fi
  local node_major
  node_major="$(node -p 'process.versions.node.split(".")[0]')"
  (( node_major >= 20 )) || die "нужен Node.js >= 20 (найден $(node -v))"
  local cargo_version="—"
  if command -v cargo >/dev/null 2>&1; then cargo_version="$(cargo --version | awk '{print $2}')"; fi
  info "node $(node -v), npm $(npm -v), $(python3 -V 2>&1), cargo $cargo_version"
  if command -v nvidia-smi >/dev/null 2>&1; then
    info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ', ' -)"
  else
    warn "nvidia-smi не найден: CUDA-путь не проверить; сборка и модели всё равно возможны"
  fi
  info "каталоги: venv=$VENV models=$MODELS cache=$CACHE"
}

has_nvenc() { [[ -x "$1" ]] && "$1" -hide_banner -encoders 2>/dev/null | grep -qi nvenc; }

detect_ffmpeg() {
  FFMPEG_GPU=""
  local candidate
  for candidate in "${GPUTOOL_FFMPEG_GPU:-}" /home/main/test/bin/ffmpeg-nvenc /usr/local/bin/ffmpeg-nvenc /opt/ffmpeg/bin/ffmpeg; do
    [[ -n "$candidate" ]] || continue
    if has_nvenc "$candidate"; then FFMPEG_GPU="$candidate"; break; fi
  done
  if [[ -z "$FFMPEG_GPU" ]] && has_nvenc "$(command -v ffmpeg)"; then
    FFMPEG_GPU="$(command -v ffmpeg)"
  fi
}

detect_ffprobe() {
  FFPROBE_GPU="${GPUTOOL_FFPROBE_GPU:-}"
  if [[ -z "$FFPROBE_GPU" && -n "${FFMPEG_GPU:-}" ]]; then
    local dir
    dir="$(dirname "$FFMPEG_GPU")"
    if [[ -x "$dir/ffprobe-nvenc" ]]; then FFPROBE_GPU="$dir/ffprobe-nvenc"; fi
  fi
}

detect_uv() {
  UV_BIN=""
  if [[ -n "${GPUTOOL_UV:-}" && -x "${GPUTOOL_UV}" ]]; then
    UV_BIN="$GPUTOOL_UV"
  elif command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  elif [[ -f "$SETTINGS" ]]; then
    UV_BIN="$(python3 - "$SETTINGS" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("uvPath", ""))
except Exception:
    print("")
PY
)"
    [[ -x "$UV_BIN" ]] || UV_BIN=""
  fi
}

# ---------------------------------------------------------------- сборка

build_project() {
  say "Сборка: npm install → npm run build → npm run build:rust"
  run npm install --cache "$ROOT/.npm-cache"
  run npm run build
  run npm run build:rust
}

# ---------------------------------------------------------------- GPU-venv

pip_install() {
  if [[ -n "$UV_BIN" ]]; then
    run "$UV_BIN" pip install --python "$PY" "$@"
  else
    run "$PY" -m pip install "$@"
  fi
}

install_runtime() {
  say "GPU-venv: $VENV"
  run mkdir -p "$GPUTOOL_HOME" "$CACHE"
  if [[ ! -x "$PY" ]]; then
    if [[ -n "$UV_BIN" ]]; then
      run "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV"
    else
      run python3 -m venv "$VENV"
      run "$PY" -m pip install --upgrade pip
    fi
  fi
  pip_install --upgrade torch torchvision --index-url "$TORCH_INDEX"
  pip_install --upgrade onnxruntime-gpu ultralytics opencv-python-headless numpy "lap>=0.5.12"
  if (( ! DRY_RUN )); then
    "$PY" - <<'PY'
import torch, onnxruntime
print(f"    torch {torch.__version__}, cuda={torch.cuda.is_available()}, onnxruntime {onnxruntime.__version__}")
PY
  fi
}

# ---------------------------------------------------------------- веса

model_ready() { [[ -f "$1" && $FORCE_MODELS -eq 0 ]]; }

ensure_repo() { # dir url
  if [[ -d "$1/.git" ]]; then
    info "репозиторий уже есть: $1"
  else
    run git clone --depth 1 "$2" "$1"
  fi
}

download_file() { # url out
  if [[ -f "$2" && $FORCE_MODELS -eq 0 ]]; then
    info "файл уже есть: $2"
  else
    run curl -fL --retry 3 -o "$2" "$1"
  fi
}

ensure_gdown() {
  if (( DRY_RUN )); then return 0; fi
  if "$PY" -c 'import gdown' >/dev/null 2>&1; then return 0; fi
  say "Ставлю gdown для Google Drive"
  pip_install --upgrade gdown
}

drive_download() { # file-id out
  ensure_gdown
  run "$PY" -m gdown "https://drive.google.com/uc?id=$1" -O "$2"
}

unzip_to() { # zip dir
  run mkdir -p "$2"
  if command -v unzip >/dev/null 2>&1; then
    run unzip -qo "$1" -d "$2"
  else
    run "$PY" -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$1" "$2"
  fi
}

first_match() { find "$1" -type f -name "$2" -print -quit 2>/dev/null | head -1; }

export_model() { # script args...
  local script="$1"; shift
  run "$PY" "$ROOT/scripts/$script" "$@"
}

install_models() {
  [[ "$MODELS_MODE" == "none" ]] && { say "Веса: пропущено (--models none)"; return 0; }
  if (( ! DRY_RUN )) && [[ ! -x "$PY" ]]; then
    die "нет $PY: сначала выполните ./install.sh без --skip-runtime"
  fi
  say "Веса: $MODELS_MODE → $MODELS"
  run mkdir -p "$MODELS/ball" "$MODELS/upscale"

  # BlurBall — лучший вариант для настольного тенниса, веса в репозитории.
  if model_ready "$MODELS/ball/blurball_tt.onnx"; then
    info "blurball_tt.onnx уже есть"
  else
    ensure_repo "$CACHE/blurball" "$BLURBALL_REPO"
    export_model export_blurball_onnx.py \
      --repo "$CACHE/blurball" --weights "$CACHE/blurball/best_model" \
      --output "$MODELS/ball/blurball_tt.onnx"
  fi

  # WASB-SBDT (теннис, MIT): веса лежат отдельным файлом в репозитории проекта.
  if model_ready "$MODELS/ball/wasb_tennis.onnx"; then
    info "wasb_tennis.onnx уже есть"
  else
    download_file "$WASB_WEIGHTS_URL" "$MODELS/ball/wasb_tennis_best.pth.tar"
    ensure_repo "$CACHE/wasb-sbdt" "$WASB_REPO"
    export_model export_wasb_onnx.py \
      --repo "$CACHE/wasb-sbdt" --weights "$MODELS/ball/wasb_tennis_best.pth.tar" \
      --output "$MODELS/ball/wasb_tennis.onnx"
  fi

  # SPAN x2 (апскейл): чекпойнты в архиве Google Drive, нужны ch48 и ch52.
  if model_ready "$MODELS/upscale/span2x_ch48_fp16.onnx" && model_ready "$MODELS/upscale/span2x_ch52_fp16.onnx"; then
    info "span2x ch48/ch52 уже есть"
  else
    ensure_repo "$CACHE/SPAN" "$SPAN_REPO"
    drive_download "$SPAN_CKPTS_ID" "$CACHE/span_ckpts.zip"
    unzip_to "$CACHE/span_ckpts.zip" "$CACHE/span_ckpt"
    local ch48 ch52
    ch48="$(first_match "$CACHE/span_ckpt" spanx2_ch48.pth)"
    ch52="$(first_match "$CACHE/span_ckpt" spanx2_ch52.pth)"
    if (( ! DRY_RUN )); then
      [[ -n "$ch48" ]] || die "в архиве SPAN не найден spanx2_ch48.pth"
      [[ -n "$ch52" ]] || die "в архиве SPAN не найден spanx2_ch52.pth"
    fi
    export_model export_span2x_onnx.py --repo "$CACHE/SPAN" --weights "$ch48" \
      --output "$MODELS/upscale/span2x_ch48_fp16.onnx"
    export_model export_span2x_onnx.py --repo "$CACHE/SPAN" --weights "$ch52" \
      --output "$MODELS/upscale/span2x_ch52_fp16.onnx"
  fi

  # TrackNetV3 — только с --models all: официальные веса бадминтона лежат
  # в архиве Google Drive (TrackNetV3_ckpts.zip).
  if [[ "$MODELS_MODE" == "all" ]]; then
    if model_ready "$MODELS/ball/tracknetv3_badminton.onnx"; then
      info "tracknetv3_badminton.onnx уже есть"
    else
      ensure_repo "$CACHE/tracknetv3" "$TRACKNETV3_REPO"
      drive_download "$TRACKNETV3_CKPTS_ID" "$CACHE/tracknetv3_ckpts.zip"
      unzip_to "$CACHE/tracknetv3_ckpts.zip" "$CACHE/tracknetv3_ckpts"
      local tracknet
      tracknet="$(first_match "$CACHE/tracknetv3_ckpts" TrackNet_best.pt)"
      if (( ! DRY_RUN )); then
        [[ -n "$tracknet" ]] || die "в архиве TrackNetV3 не найден TrackNet_best.pt"
      fi
      export_model export_tracknetv3_onnx.py --repo "$CACHE/tracknetv3" \
        --weights "$tracknet" --output "$MODELS/ball/tracknetv3_badminton.onnx"
    fi
  fi
}

# ---------------------------------------------------------------- manifest

write_manifest() {
  (( DO_MANIFEST )) || { info "manifest.json: пропущено"; return 0; }
  if [[ ! -d "$MODELS" ]]; then
    info "каталога $MODELS нет: manifest.json не нужен"
    return 0
  fi
  if (( DRY_RUN )); then
    printf '    [dry-run] manifest.json по %s\n' "$MODELS"
    return 0
  fi
  python3 - "$MODELS" <<'PY'
import hashlib, json, os, sys
root = os.path.abspath(sys.argv[1])
items = []
for base, _dirs, names in os.walk(root):
    for name in sorted(names):
        if name == "manifest.json":
            continue
        path = os.path.join(base, name)
        rel = os.path.relpath(path, root)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        items.append({"path": rel, "bytes": os.path.getsize(path), "sha256": digest.hexdigest()})
items.sort(key=lambda item: item["path"])
manifest = {"root": root, "items": items}
tmp = os.path.join(root, "manifest.json.tmp")
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
os.replace(tmp, os.path.join(root, "manifest.json"))
print(f"    manifest: {len(items)} файлов")
PY
}

verify_models() {
  local manifest="$MODELS/manifest.json"
  [[ -f "$manifest" ]] || die "нет $manifest: сначала выполните ./install.sh"
  python3 - "$manifest" <<'PY'
import hashlib, json, os, sys
manifest_path = sys.argv[1]
data = json.load(open(manifest_path, encoding="utf-8"))
root = data["root"]
bad = []
for item in data["items"]:
    path = os.path.join(root, item["path"])
    if not os.path.isfile(path):
        bad.append(f"нет файла: {item['path']}")
        continue
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != item["sha256"]:
        bad.append(f"sha256 не совпал: {item['path']}")
if bad:
    print("\n".join(bad), file=sys.stderr)
    raise SystemExit(1)
print(f"    модели в порядке: {len(data['items'])} файлов")
PY
}

# ---------------------------------------------------------------- настройки

update_settings() {
  run mkdir -p "$GPUTOOL_HOME"
  local ball="" upscale="" python_value=""
  [[ -x "$PY" ]] && python_value="$PY"
  [[ -f "$MODELS/ball/blurball_tt.onnx" ]] && ball="$MODELS/ball/blurball_tt.onnx"
  [[ -f "$MODELS/upscale/span2x_ch48_fp16.onnx" ]] && upscale="$MODELS/upscale/span2x_ch48_fp16.onnx"
  say "Настройки: $SETTINGS"
  if (( DRY_RUN )); then
    printf '    [dry-run] pythonPath=%s\n' "${python_value:-<venv не создан>}"
    [[ -n "$ball" ]] && printf '    [dry-run] ballModelPath=%s\n' "$ball"
    [[ -n "$upscale" ]] && printf '    [dry-run] upscaleModelPath=%s\n' "$upscale"
    return 0
  fi
  python3 - "$SETTINGS" "$python_value" "$ball" "$upscale" "${FFMPEG_GPU:-}" "${FFPROBE_GPU:-}" <<'PY'
import json, os, sys

path, python, ball, upscale, ffmpeg_gpu, ffprobe_gpu = sys.argv[1:7]
data = {}
if os.path.isfile(path):
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        data = {}
changed = {}


def set_value(key, value, force=False):
    if not value:
        return
    current = data.get(key)
    if force or not current or current in ("python3", "/usr/bin/python3"):
        data[key] = value
        changed[key] = value


set_value("pythonPath", python, force=True)
set_value("ballModelPath", ball)
set_value("upscaleModelPath", upscale)
set_value("ffmpegGpuPath", ffmpeg_gpu)
set_value("ffprobeGpuPath", ffprobe_gpu)

os.makedirs(os.path.dirname(path), exist_ok=True)
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(data, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
os.replace(tmp, path)
print("    обновлено:", ", ".join(changed) if changed else "без изменений")
PY
}

# ---------------------------------------------------------------- main

preflight
detect_ffmpeg
detect_ffprobe
detect_uv
[[ -n "${FFMPEG_GPU:-}" ]] && info "GPU-ffmpeg: $FFMPEG_GPU" || warn "GPU-ffmpeg не найден; doctor покажет обычный ffmpeg"
[[ -n "${UV_BIN:-}" ]] && info "uv: $UV_BIN" || info "uv не найден: venv через python3 -m venv"

if (( VERIFY )); then
  verify_models
  exit 0
fi

say "План: build=$DO_BUILD runtime=$DO_RUNTIME models=$MODELS_MODE manifest=$DO_MANIFEST"
if (( CHECK )); then
  info "check: окружение в порядке; запустите ./install.sh"
  exit 0
fi

(( DO_BUILD )) && build_project
(( DO_RUNTIME )) && install_runtime
install_models
write_manifest
update_settings

say "Готово: версия $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$ROOT/package.json" 2>/dev/null || echo '?')"
info "плагин ставится через менеджер плагинов DSH (install_bundle с путём $ROOT),"
info "профиль вручную не правится; после установки перезапустите DSH и откройте панель GPU tool."
info "проверка моделей: ./install.sh --verify-models"
