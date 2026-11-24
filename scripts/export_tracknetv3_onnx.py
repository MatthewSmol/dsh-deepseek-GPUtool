#!/usr/bin/env python3
"""Экспорт чекпойнта TrackNetV3 (та же архитектура, что в wasn-lab/TrackNetV3_TableTennis).

Модель `TrackNet` уже содержит sigmoid на выходе: получаем вероятностные heatmap
[1, seq_len, 288, 512] из входа [1, seq_len*3, 288, 512]. Скрипт не копирует
архитектуру в плагин, а импортирует `model.py` из клона репозитория.

Пример:
  python scripts/export_tracknetv3_onnx.py \
    --repo ~/.cache/gputool/tracknetv3-tt \
    --weights ~/.dsh/gputool/models/ball/TrackNet_best.pt \
    --output ~/.dsh/gputool/models/ball/tracknetv3_tt.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import torch


def load_tracknet(repo: Path):
    module_path = repo / "model.py"
    if not module_path.is_file():
        raise SystemExit(f"model.py не найден: {module_path} (укажите --repo на клон TrackNetV3_TableTennis)")
    spec = importlib.util.spec_from_file_location("tracknet_model", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.TrackNet


def input_channels(seq_len: int, bg_mode: str) -> int:
    if bg_mode == "subtract":
        return seq_len
    if bg_mode == "subtract_concat":
        return seq_len * 4
    if bg_mode == "concat":
        return (seq_len + 1) * 3
    return seq_len * 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path.home() / ".cache/gputool/tracknetv3-tt"))
    parser.add_argument("--weights", help="чекпойнт TrackNet_best.pt")
    parser.add_argument("--random-init", action="store_true", help="экспорт случайной модели (проверка пайплайна)")
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--bg-mode", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    repo = Path(args.repo).expanduser()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    TrackNet = load_tracknet(repo)

    seq_len = args.seq_len
    bg_mode = args.bg_mode
    state = None
    if args.weights and not args.random_init:
        checkpoint = torch.load(Path(args.weights).expanduser(), map_location="cpu", weights_only=False)
        params = checkpoint.get("param_dict") or {}
        seq_len = int(params.get("seq_len", seq_len))
        bg_mode = str(params.get("bg_mode", bg_mode))
        state = checkpoint.get("model") or checkpoint

    channels = input_channels(seq_len, bg_mode)
    model = TrackNet(in_dim=channels, out_dim=seq_len).eval()
    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("random-init: экспортирую модель без весов (только проверка формы)")

    dummy = torch.zeros(1, channels, args.height, args.width)
    with torch.no_grad():
        exported = model(dummy)
    print(f"torch: in={tuple(dummy.shape)} out={tuple(exported.shape)} seq_len={seq_len} bg={bg_mode!r}")

    torch.onnx.export(
        model,
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
