#!/usr/bin/env python3
"""Экспорт SPAN x2 (Swift Parameter-free Attention Network) в ONNX fp16.

Веса — официальные чекпойнты SPAN (Apache-2.0), архив со всеми вариантами:
https://drive.google.com/file/d/1iYUA2TzKuxI0vzmA-UXr_nB43XgPOXUg/view
Нужны только `spanx2_ch48.pth` / `spanx2_ch52.pth`; архитектура берётся из
клона `hongyuanyu/SPAN` (файл `basicsr/archs/span_arch.py`), без установки BasicSR.

Пример:
  python scripts/export_span2x_onnx.py \
    --repo ~/.cache/gputool/SPAN \
    --weights ~/.cache/gputool/span/ckpt/spanx2_ch48.pth \
    --output ~/.dsh/gputool/models/upscale/span2x_ch48_fp16.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import torch


def load_span(repo: Path):
    """Импортирует SPAN, подменяя basicsr.registry заглушкой (BasicSR не нужен)."""
    for name in ("basicsr", "basicsr.utils", "basicsr.utils.registry"):
        sys.modules.setdefault(name, types.ModuleType(name))

    class Registry:
        def register(self, obj=None, **kwargs):
            def decorate(cls):
                return cls
            return decorate if obj is None else obj

    sys.modules["basicsr.utils.registry"].ARCH_REGISTRY = Registry()
    module_path = repo / "basicsr" / "archs" / "span_arch.py"
    if not module_path.is_file():
        raise SystemExit(f"span_arch.py не найден: {module_path} (укажите --repo на клон SPAN)")
    spec = importlib.util.spec_from_file_location("span_arch", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SPAN


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path.home() / ".cache/gputool/SPAN"))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--channels", type=int, default=48, choices=[48, 52])
    parser.add_argument("--scale", type=int, default=2, choices=[2])
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    repo = Path(args.repo).expanduser()
    weights = Path(args.weights).expanduser()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    SPAN = load_span(repo)
    model = SPAN(3, 3, feature_channels=args.channels, upscale=args.scale).eval()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint.get("params_ema") or checkpoint.get("params") or checkpoint.get("model") or checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:5])

    model = model.half().eval()
    dummy = torch.zeros(1, 3, args.height, args.width, dtype=torch.float16)
    with torch.no_grad():
        result = model(dummy)
    print("torch output:", tuple(result.shape))

    torch.onnx.export(
        model,
        dummy,
        str(output),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "N", 2: "H", 3: "W"}, "output": {0: "N", 2: "H2", 3: "W2"}},
        dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    print("onnx inputs:", [(item.name, item.shape, item.type) for item in session.get_inputs()])
    print("onnx outputs:", [(item.name, item.shape, item.type) for item in session.get_outputs()])
    print("onnx bytes:", output.stat().st_size)
    print("onnx sha256:", sha256(output))


if __name__ == "__main__":
    main()
