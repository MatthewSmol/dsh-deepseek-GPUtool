#!/usr/bin/env python3
"""Скан лиц YuNet + SFace на CUDA (ONNX Runtime) для сервиса GPU tool.

Воркер декодирует кадры через ffmpeg (rawvideo bgr24 в stdout), ищет лица YuNet
и, если задано эталонное изображение, выравнивает лицо и сравнивает эмбеддинг
SFace с эталоном по косинусной близости.

Конфиг:
  source, ffmpegPath, fps, startSec, durationSec, width, height, totalFrames,
  modelDir, refImage?, conf, threshold, matchThreshold, outputDir, device
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np


def emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def fail(code, message, details=None):
    emit({"type": "error", "code": code, "message": message, "details": details})
    sys.exit(1)


REFERENCE_POINTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceEngine:
    def __init__(self, model_dir, providers, require_gpu):
        import onnxruntime as ort

        self.ort = ort
        self.yunet_path = os.path.join(model_dir, "yunet.onnx")
        self.sface_path = os.path.join(model_dir, "sface.onnx")
        for path in (self.yunet_path, self.sface_path):
            if not os.path.exists(path):
                fail("face-model-missing", "model not found: %s" % path)
        available = ort.get_available_providers()
        selected = [name for name in providers if name in available]
        if "CPUExecutionProvider" not in selected:
            selected.append("CPUExecutionProvider")
        self.det = ort.InferenceSession(self.yunet_path, providers=selected)
        self.rec = ort.InferenceSession(self.sface_path, providers=selected)
        used = self.det.get_providers()
        if require_gpu and not any(name.startswith(("CUDA", "Tensorrt")) for name in used):
            fail("cuda-ep-unavailable", "face models fell back to CPU; GPU is required", {"used": used})
        self.det_input = self.det.get_inputs()[0]
        shape = self.det_input.shape
        # Динамические размеры приходят как -1 или строка; берём 320x320.
        self.in_w = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else 320
        self.in_h = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 320
        self.outputs = {output.name: output for output in self.det.get_outputs()}
        self.providers_used = used

    def anchors(self, stride):
        cols = self.in_w // stride
        rows = self.in_h // stride
        grid = np.zeros((rows * cols, 2), dtype=np.float32)
        for row in range(rows):
            grid[row * cols : (row + 1) * cols, 0] = np.arange(cols)
            grid[row * cols : (row + 1) * cols, 1] = row
        return grid

    def detect(self, frame, conf):
        rgb = cv2.resize(frame, (self.in_w, self.in_h))
        blob = cv2.dnn.blobFromImage(rgb, 1.0, (self.in_w, self.in_h), (104.0, 117.0, 123.0), swapRB=False)
        outputs = self.det.run(None, {self.det_input.name: blob})
        names = [output.name for output in self.det.get_outputs()]
        by_suffix = {}
        for name, value in zip(names, outputs):
            parts = name.split("_")
            if not parts[-1].isdigit():
                continue
            stride = int(parts[-1])
            by_suffix.setdefault(stride, {})["".join(parts[:-1]).lower()] = np.asarray(value)
        boxes = []
        scores = []
        landmarks = []
        for stride, group in by_suffix.items():
            cls = group.get("cls")
            obj = group.get("obj")
            bbox = group.get("bbox")
            kps = group.get("kps")
            if cls is None or obj is None or bbox is None or kps is None:
                continue
            cls = cls.reshape(-1)
            obj = obj.reshape(-1, 1)
            bbox = bbox.reshape(-1, 4)
            kps = kps.reshape(-1, 10)
            grid = self.anchors(stride)
            count = min(len(cls), len(obj), len(bbox), len(kps), len(grid))
            if count == 0:
                continue
            cls = cls[:count]
            obj = obj[:count, 0]
            bbox = bbox[:count]
            kps = kps[:count]
            grid = grid[:count]
            score = np.sqrt(np.clip(cls * obj, 0.0, 1.0))
            keep = score >= conf
            if not np.any(keep):
                continue
            grid = grid[keep]
            bbox = bbox[keep]
            kps = kps[keep]
            score = score[keep]
            cx = (grid[:, 0] + bbox[:, 0]) * stride
            cy = (grid[:, 1] + bbox[:, 1]) * stride
            width = np.exp(bbox[:, 2]) * stride
            height = np.exp(bbox[:, 3]) * stride
            for index in range(len(score)):
                x = cx[index] - width[index] / 2.0
                y = cy[index] - height[index] / 2.0
                boxes.append([float(x), float(y), float(width[index]), float(height[index])])
                scores.append(float(score[index]))
                landmarks.append(
                    [
                        [float((grid[index, 0] + kps[index, 2 * k]) * stride), float((grid[index, 1] + kps[index, 2 * k + 1]) * stride)]
                        for k in range(5)
                    ]
                )
        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf, 0.3)
        result = []
        for index in np.array(indices).reshape(-1):
            x, y, w, h = boxes[int(index)]
            result.append(
                {
                    "score": round(scores[int(index)], 4),
                    "box": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "landmarks": [[round(px, 2), round(py, 2)] for px, py in landmarks[int(index)]],
                }
            )
        return result

    def align(self, frame, landmarks):
        src = np.array(landmarks, dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(src, REFERENCE_POINTS, method=cv2.LMEDS)
        if matrix is None:
            return None
        return cv2.warpAffine(frame, matrix, (112, 112), borderValue=0.0)

    def embed(self, aligned):
        blob = ((aligned.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis, ...]
        output = self.rec.run(None, {self.rec.get_inputs()[0].name: blob})[0]
        vector = np.asarray(output).reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


def reference_embedding(engine, image_path, conf):
    image = cv2.imread(image_path)
    if image is None:
        fail("face-ref-missing", "cannot read reference image: %s" % image_path)
    faces = engine.detect(image, conf)
    if not faces:
        fail("face-ref-no-face", "YuNet found no face in the reference image")
    face = max(faces, key=lambda item: item["box"][2] * item["box"][3])
    aligned = engine.align(image, face["landmarks"])
    if aligned is None:
        fail("face-align-failed", "cannot align the reference face")
    return engine.embed(aligned)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config)

    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:
        fail("onnx-deps-missing", "onnxruntime is not installed: %s" % exc)

    source = cfg.get("source")
    if not source or not os.path.exists(source):
        fail("input-missing", "source not found: %s" % source)
    model_dir = cfg.get("modelDir")
    if not model_dir:
        fail("invalid-args", "modelDir is required")
    device = str(cfg.get("device") or "cuda:0")
    require_gpu = device != "cpu"
    providers = cfg.get("providers") or ["CUDAExecutionProvider", "CPUExecutionProvider"]

    engine = FaceEngine(model_dir, providers, require_gpu)
    emit({"type": "log", "message": "providers: %s" % ", ".join(engine.providers_used)})

    ref = None
    if cfg.get("refImage"):
        ref = reference_embedding(engine, cfg["refImage"], float(cfg.get("conf") or 0.6))
        emit({"type": "log", "message": "reference embedding ready"})

    output_dir = cfg.get("outputDir") or os.path.join(os.getcwd(), "face-out")
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "face.jsonl")

    width = int(cfg.get("width") or 1280)
    height = int(cfg.get("height") or 720)
    fps = float(cfg.get("fps") or 1.0)
    duration = cfg.get("durationSec")
    total = cfg.get("totalFrames")
    if total is None and duration:
        total = max(1, int(float(duration) * fps))
    ffmpeg = cfg.get("ffmpegPath")
    if not ffmpeg:
        fail("invalid-args", "ffmpegPath is required")

    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if cfg.get("startSec"):
        command += ["-ss", str(cfg["startSec"])]
    command += ["-i", source]
    if duration:
        command += ["-t", str(duration)]
    command += ["-vf", "fps=%s" % fps, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    emit({"type": "log", "message": "decode: " + " ".join(command)})

    frame_bytes = width * height * 3
    started = time.time()
    frames = 0
    faces_total = 0
    matches = 0
    match_threshold = float(cfg.get("matchThreshold") or 0.363)
    conf = float(cfg.get("conf") or 0.6)
    last_emit = 0.0

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with open(jsonl_path, "w", encoding="utf-8") as handle:
            while True:
                buffer = process.stdout.read(frame_bytes)
                if not buffer or len(buffer) < frame_bytes:
                    break
                frame = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3))
                frames += 1
                faces = engine.detect(frame, conf)
                rows = []
                for face in faces:
                    row = dict(face)
                    if ref is not None:
                        aligned = engine.align(frame, face["landmarks"])
                        if aligned is not None:
                            embedding = engine.embed(aligned)
                            similarity = float(np.dot(ref, embedding))
                            row["sim"] = round(similarity, 4)
                            row["match"] = similarity >= match_threshold
                            if row["match"]:
                                matches += 1
                    rows.append(row)
                faces_total += len(faces)
                handle.write(json.dumps({"frame": frames, "time": round(frames / fps, 3), "faces": rows}, ensure_ascii=False) + "\n")
                now = time.time()
                if now - last_emit >= 0.4:
                    last_emit = now
                    elapsed = max(1e-6, now - started)
                    rate = frames / elapsed
                    percent = max(0.0, min(100.0, frames * 100.0 / total)) if total else None
                    eta = int(max(0.0, ((total or 0) - frames) / max(rate, 1e-6)) * 1000) if total else None
                    emit(
                        {
                            "type": "progress",
                            "phase": "faces",
                            "percent": percent,
                            "frame": frames,
                            "totalFrames": total,
                            "fps": round(rate, 2),
                            "rate": round(rate, 3),
                            "etaMs": eta,
                        }
                    )
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", "replace").strip()
        process.stderr.close()
        process.wait(timeout=10)
    if stderr:
        emit({"type": "log", "message": "ffmpeg: " + stderr[-2000:]})

    emit(
        {
            "type": "result",
            "value": {
                "frames": frames,
                "faces": faces_total,
                "matches": matches,
                "jsonl": jsonl_path,
                "providers": engine.providers_used,
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
        fail("face-failed", str(exc), {"traceback": traceback.format_exc()[-4000:]})
