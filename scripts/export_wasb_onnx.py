#!/usr/bin/env python3
"""Экспорт весов WASB-SBDT (теннис/настольный теннис) в ONNX.

Модель — HRNet из репозитория nttcom/WASB-SBDT (MIT). Скрипт не копирует код
модели в плагин: он импортирует `src/models/hrnet.py` из клона репозитория,
строит конфиг из `wasb.yaml` (значения зафиксированы ниже) и экспортирует
ONNX-граф с фиксированным входом [1, 9, 288, 512].

Пример:
  python scripts/export_wasb_onnx.py \
    --repo ~/.cache/gputool/wasb-sbdt \
    --weights ~/.dsh/gputool/models/ball/wasb_tennis_best.pth.tar \
    --output ~/.dsh/gputool/models/ball/wasb_tennis.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

import torch


class Config(dict):
    """Словарь с доступом через атрибуты: HRNet читает и cfg['MODEL'], и cfg.MODEL.EXTRA."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:  # pragma: no cover - защита от опечаток
            raise AttributeError(name) from error


def deep_config(value):
    if isinstance(value, dict):
        return Config({key: deep_config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [deep_config(item) for item in value]
    return value


# Значения из src/configs/model/wasb.yaml (коммит 923462cacdeb3353b84ddebdedb3f4b7a8553b0f).
WASB_CONFIG = {
    "name": "hrnet",
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
            "STAGE1": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 1,
                "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [1],
                "NUM_CHANNELS": [32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2],
                "NUM_CHANNELS": [16, 32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64],
                "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64, 128],
                "FUSE_METHOD": "SUM",
            },
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
        },
        "INIT_WEIGHTS": True,
    },
}


def load_hrnet(repo: Path):
    module_path = repo / "src" / "models" / "hrnet.py"
    if not module_path.is_file():
        raise SystemExit(f"hrnet.py не найден: {module_path} (укажите --repo на клон WASB-SBDT)")
    spec = importlib.util.spec_from_file_location("wasb_hrnet", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.HRNet


class HeatmapOutput(torch.nn.Module):
    """HRNet возвращает dict {scale: tensor}; для ONNX нужен один тензор."""

    def __init__(self, model: torch.nn.Module, scale: int = 0):
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, x):  # noqa: D102
        return self.model(x)[self.scale]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path.home() / ".cache/gputool/wasb-sbdt"))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    repo = Path(args.repo).expanduser()
    weights = Path(args.weights).expanduser()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    HRNet = load_hrnet(repo)
    cfg = deep_config(WASB_CONFIG)
    model = HRNet(cfg).eval()

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  missing:", missing[:5])
        if unexpected:
            print("  unexpected:", unexpected[:5])

    wrapped = HeatmapOutput(model, scale=0).eval()
    dummy = torch.zeros(1, 3 * int(cfg["frames_in"]), args.height, args.width)
    with torch.no_grad():
        exported = wrapped(dummy)
    print("torch output:", tuple(exported.shape))

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
    inputs = [(item.name, item.shape) for item in session.get_inputs()]
    outputs = [(item.name, item.shape) for item in session.get_outputs()]
    print("onnx inputs:", inputs)
    print("onnx outputs:", outputs)
    print("onnx bytes:", output.stat().st_size)
    print("onnx sha256:", sha256(output))


if __name__ == "__main__":
    main()
