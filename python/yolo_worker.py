#!/usr/bin/env python3
"""Запуск YOLO11 для MCP-сервиса GPU tool.

Читает JSON-конфиг из --config, выполняет predict/track ultralytics на CUDA
(или CPU, если это явно запрошено) и построчно отдаёт JSONL-события прогресса
и результата в stdout.
"""
import argparse
import json
import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import sys
import time
import traceback


def emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def fail(code, message, details=None):
    emit({"type": "error", "code": code, "message": message, "details": details})
    sys.exit(1)


def center_points(result, rows):
    """Центр bbox человека вместо 17 невидимых точек скелета (YOLO11n-pose)."""
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return rows
    try:
        data = keypoints.xy.cpu().numpy() if hasattr(keypoints.xy, "cpu") else keypoints.xy
        conf = keypoints.conf
        conf_data = conf.cpu().numpy() if conf is not None and hasattr(conf, "cpu") else conf
    except Exception:
        return rows
    for index, row in enumerate(rows):
        if index >= len(data):
            break
        points = data[index]
        visible = [p for p in range(len(points)) if conf_data is None or (p < len(conf_data[index]) and conf_data[index][p] > 0.3)]
        if visible:
            xs = [float(points[p][0]) for p in visible]
            ys = [float(points[p][1]) for p in visible]
            row["center"] = [round(sum(xs) / len(xs), 2), round(sum(ys) / len(ys), 2)]
            row["keypointsVisible"] = len(visible)
        else:
            x1, y1, x2, y2 = row["box"]
            row["center"] = [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)]
            row["keypointsVisible"] = 0
    return rows


def count_total(source):
    try:
        import cv2

        if os.path.isdir(source):
            total = 0
            for _root, _dirs, files in os.walk(source):
                total += sum(1 for name in files if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
            return total or None
        if os.path.isfile(source):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
                return total or None
    except Exception:
        return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config)

    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - environment dependent
        fail("yolo-deps-missing", "ultralytics/torch are not available: %s" % exc)

    device = str(cfg.get("device") or "cuda:0")
    use_cuda = device != "cpu"
    if use_cuda and not torch.cuda.is_available():
        fail(
            "yolo-gpu-unavailable",
            "CUDA is not available to this process; pass device='cpu' to run YOLO on the CPU",
            {"torch": torch.__version__, "device": device},
        )

    model_path = cfg.get("model")
    source = cfg.get("source")
    if not model_path:
        fail("invalid-args", "model is required")
    if not source:
        fail("invalid-args", "source is required")
    if not os.path.exists(model_path):
        fail("yolo-model-missing", "model file not found: %s" % model_path)
    if not os.path.exists(source):
        fail("input-missing", "source not found: %s" % source)

    output_dir = cfg.get("outputDir") or os.path.join(os.getcwd(), "yolo-out")
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "yolo.jsonl")
    total = count_total(source)
    if cfg.get("maxFrames"):
        total = min(total, int(cfg["maxFrames"])) if total else int(cfg["maxFrames"])

    emit({"type": "log", "message": "loading %s on %s" % (model_path, device)})
    emit({"type": "progress", "phase": "load", "percent": 0, "totalFrames": total})
    try:
        model = YOLO(model_path)
    except Exception as exc:
        fail("yolo-load-failed", "cannot load model: %s" % exc)

    target = device
    if use_cuda and ":" not in target:
        target = "cuda:0"

    kwargs = {
        "source": source,
        "stream": True,
        "verbose": False,
        "device": target,
        "imgsz": int(cfg.get("imgsz") or 640),
        "conf": float(cfg.get("conf") or 0.25),
        "iou": float(cfg.get("iou") or 0.45),
    }
    if cfg.get("classes"):
        kwargs["classes"] = [int(value) for value in cfg["classes"]]
    if use_cuda and cfg.get("half", True):
        kwargs["half"] = True
    if cfg.get("saveAnnotated"):
        kwargs.update({"save": True, "project": output_dir, "name": "annotated", "exist_ok": True})

    started = time.time()
    frames = 0
    detections = 0
    track_ids = set()
    last_emit = 0.0
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        try:
            if cfg.get("track"):
                kwargs["tracker"] = cfg.get("tracker") or "botsort.yaml"
                kwargs["persist"] = True
                results = model.track(**kwargs)
            else:
                results = model.predict(**kwargs)
            for result in results:
                frames += 1
                rows = []
                boxes = getattr(result, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
                    classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
                    ids = boxes.id.cpu().numpy() if getattr(boxes, "id", None) is not None else None
                    for index in range(len(xyxy)):
                        box = [round(float(v), 2) for v in xyxy[index].tolist()]
                        class_id = int(classes[index])
                        row = {
                            "box": box,
                            "conf": round(float(confs[index]), 4),
                            "class": class_id,
                            "name": result.names.get(class_id, str(class_id)) if isinstance(result.names, dict) else str(class_id),
                        }
                        if ids is not None:
                            row["trackId"] = int(ids[index])
                            track_ids.add(int(ids[index]))
                        rows.append(row)
                        detections += 1
                record = {"frame": frames, "detections": center_points(result, rows)}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                now = time.time()
                if now - last_emit >= 0.4 or (total and frames >= total):
                    last_emit = now
                    elapsed = max(1e-6, now - started)
                    fps = frames / elapsed
                    percent = None
                    eta = None
                    if total:
                        percent = max(0.0, min(100.0, frames * 100.0 / total))
                        eta = int(max(0.0, (total - frames) / max(fps, 1e-6)) * 1000)
                    emit(
                        {
                            "type": "progress",
                            "phase": "infer",
                            "percent": percent,
                            "frame": frames,
                            "totalFrames": total,
                            "fps": round(fps, 2),
                            "rate": round(fps, 3),
                            "etaMs": eta,
                        }
                    )
                if cfg.get("maxFrames") and frames >= int(cfg["maxFrames"]):
                    break
        except Exception as exc:
            fail("yolo-failed", str(exc), {"traceback": traceback.format_exc()[-4000:]})

    emit(
        {
            "type": "result",
            "value": {
                "frames": frames,
                "detections": detections,
                "trackIds": sorted(track_ids),
                "jsonl": jsonl_path,
                "annotatedDir": os.path.join(output_dir, "annotated") if cfg.get("saveAnnotated") else None,
                "device": device,
            },
        }
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        fail("yolo-failed", str(exc), {"traceback": traceback.format_exc()[-4000:]})
