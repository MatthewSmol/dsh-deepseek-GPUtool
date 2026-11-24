#!/usr/bin/env python3
"""Экспорт весов BlurBall (настольный теннис) в ONNX.

BlurBall — форк WASB-SBDT (HRNet, MIT): 3 кадра RGB (9 каналов) 288×512 →
3 heatmap-канала 288×512 с sigmoid. Веса лежат прямо в репозитории
`git clone https://github.com/mrzkp/blurball` (`best_model` — вариант с SE,
`blurball_best` — базовая WASB-архитектура).

Пример:
  python scripts/export_blurball_onnx.py \
    --repo ~/.cache/gputool/blurball \
    --weights ~/.cache/gputool/blurball/best_model \
    --output ~/.dsh/gputool/models/ball/blurball_tt.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import torch

BASE_CFG = {
    "frames_in": 3,
    "frames_out": 3,
    "inp_height": 288,
    "inp_width": 512,
    "out_height": 288,
    "out_width": 512,
    "rgb_diff": False,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "PRETRAINED_LAYERS": ["*"],
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK", "NUM_BLOCKS": [1], "NUM_CHANNELS": [32], "FUSE_METHOD": "SUM"},
            "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32], "FUSE_METHOD": "SUM"},
            "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64], "FUSE_METHOD": "SUM"},
            "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [16, 32, 64, 128], "FUSE_METHOD": "SUM"},
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
        },
        "INIT_WEIGHTS": True,
    },
}


class Config(dict):
    __getattr__ = lambda self, name: self[name]


def deep(value):
    if isinstance(value, dict):
        return Config({key: deep(item) for key, item in value.items()})
    if isinstance(value, list):
        return [deep(item) for item in value]
    return value


def load_class(repo: Path, module_file: str, class_name: str):
    path = repo / "src" / "models" / module_file
    if not path.is_file():
        raise SystemExit(f"не найден {path}; укажите --repo на клон blurball")
    spec = importlib.util.spec_from_file_location(f"blurball_{module_file[:-3]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, class_name)


class HeatmapOutput(torch.nn.Module):
    """BlurBall возвращает dict {scale: tensor}; ONNX ждёт один тензор."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path.home() / ".cache/gputool/blurball"))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arch", choices=["auto", "blurball", "blurball_se"], default="auto")
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    repo = Path(args.repo).expanduser()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(Path(args.weights).expanduser(), map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint

    cfg = deep(dict(BASE_CFG, name="blurball"))
    Plain = load_class(repo, "blurball.py", "BlurBall")
    SE = load_class(repo, "blurball_se.py", "BlurBall_SE")

    arch = args.arch
    if arch == "auto":
        probe = Plain(cfg).eval()
        missing, unexpected = probe.load_state_dict(state, strict=False)
        arch = "blurball_se" if unexpected and all(".se." in key for key in unexpected) else "blurball"
        print(f"auto arch: {arch} (unexpected={len(unexpected)})")

    model = (SE if arch == "blurball_se" else Plain)(cfg).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"arch={arch} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:5])

    wrapped = HeatmapOutput(model).eval()
    dummy = torch.zeros(1, 3 * int(BASE_CFG["frames_in"]), args.height, args.width)
    with torch.no_grad():
        result = wrapped(dummy)
    print("torch output:", tuple(result.shape))

    torch.onnx.export(
        wrapped,
        dummy,
        str(output),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    print("onnx inputs:", [(item.name, item.shape) for item in session.get_inputs()])
    print("onnx outputs:", [(item.name, item.shape) for item in session.get_outputs()])
    print("onnx bytes:", output.stat().st_size)
    print("onnx sha256:", sha256(output))


if __name__ == "__main__":
    main()
