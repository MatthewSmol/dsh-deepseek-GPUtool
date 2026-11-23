#!/usr/bin/env python3
"""Универсальный запуск ONNX Runtime с провайдерами CUDA/TensorRT.

Конфиг:
  model: путь к .onnx
  inputs: [{name?: str, path: str (npy)}] в порядке подачи
  providers: необязательный список провайдеров
  requireGpu: bool (по умолчанию true) — падать вместо тихого CPU
  outputDir: куда сохранять выходные .npy
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np


def emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def fail(code, message, details=None):
    emit({"type": "error", "code": code, "message": message, "details": details})
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config)

    try:
        import onnxruntime as ort
    except Exception as exc:
        fail("onnx-deps-missing", "onnxruntime is not installed: %s" % exc)

    model = cfg.get("model")
    if not model or not os.path.exists(model):
        fail("onnx-model-missing", "model file not found: %s" % model)

    available = list(ort.get_available_providers())
    requested = cfg.get("providers") or ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [name for name in requested if name in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    require_gpu = cfg.get("requireGpu", True)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    emit({"type": "progress", "phase": "load", "percent": 0})
    try:
        session = ort.InferenceSession(model, sess_options=options, providers=providers)
    except Exception as exc:
        fail("onnx-session-failed", "cannot create ONNX session: %s" % exc, {"providers": providers})

    used = session.get_providers()
    gpu_used = any(name in used for name in ("TensorrtExecutionProvider", "CUDAExecutionProvider"))
    if require_gpu and not gpu_used:
        fail(
            "cuda-ep-unavailable",
            "ONNX session fell back to CPU; GPU provider is required",
            {"available": available, "requested": providers, "used": used},
        )

    input_meta = session.get_inputs()
    input_specs = cfg.get("inputs") or []
    if not input_specs:
        fail("invalid-args", "inputs is required (list of .npy paths)")
    feed = {}
    for index, meta in enumerate(input_meta):
        spec = None
        for candidate in input_specs:
            if candidate.get("name") == meta.name:
                spec = candidate
                break
        if spec is None and index < len(input_specs):
            spec = input_specs[index]
        if spec is None:
            fail("onnx-input-missing", "no input supplied for %s" % meta.name)
        path = spec.get("path")
        if not path or not os.path.exists(path):
            fail("input-missing", "input array not found: %s" % path)
        array = np.load(path)
        dtype = meta.type
        if "float16" in dtype:
            array = array.astype(np.float16)
        elif "float" in dtype:
            array = array.astype(np.float32)
        elif "int64" in dtype:
            array = array.astype(np.int64)
        feed[meta.name] = array
        emit({"type": "log", "message": "input %s %s" % (meta.name, list(array.shape))})

    started = time.time()
    emit({"type": "progress", "phase": "infer", "percent": 10})
    try:
        outputs = session.run(None, feed)
    except Exception as exc:
        fail("onnx-run-failed", "inference failed: %s" % exc, {"used": used})
    elapsed = time.time() - started

    output_dir = cfg.get("outputDir")
    summary = []
    for index, (meta, value) in enumerate(zip(session.get_outputs(), outputs)):
        array = np.asarray(value)
        entry = {
            "name": meta.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(np.min(array)) if array.size else None,
            "max": float(np.max(array)) if array.size else None,
        }
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "output_%02d_%s.npy" % (index, meta.name.replace("/", "_")))
            np.save(path, array)
            entry["path"] = path
        summary.append(entry)

    emit({"type": "progress", "phase": "done", "percent": 100, "fps": round(1.0 / max(elapsed, 1e-6), 3)})
    emit(
        {
            "type": "result",
            "value": {
                "providers": used,
                "elapsedMs": int(elapsed * 1000),
                "outputs": summary,
            },
        }
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        fail("onnx-failed", str(exc), {"traceback": traceback.format_exc()[-4000:]})
