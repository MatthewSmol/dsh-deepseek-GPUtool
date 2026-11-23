#!/usr/bin/env python3
"""OCR табло PP-OCRv5 на CUDA (ONNX Runtime) для сервиса GPU tool.

Запускает мобильный детектор PP-OCRv5 и CTC-распознаватель из каталога моделей
ocr-ru на изображениях, выборочных кадрах видео или явных таймкодах и пишет JSONL.

Конфиг:
  modelDir, dictPath?, source? (видео), images? (пути), atSec? (список),
  fps?, startSec?, durationSec?, region? {x,y,w,h}, provider? (cuda|cpu),
  detLimit?, outputDir, width?, height?
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


class PpOcr:
    def __init__(self, model_dir, dict_path, require_gpu, providers):
        import onnxruntime as ort

        self.det_path = os.path.join(model_dir, "PP-OCRv5_mobile_det.onnx")
        self.rec_path = os.path.join(model_dir, "PP-OCRv5_mobile_rec.onnx")
        for path in (self.det_path, self.rec_path):
            if not os.path.exists(path):
                fail("ocr-model-missing", "model not found: %s" % path)
        self.dict_path = dict_path or os.path.join(model_dir, "ppocrv5_dict.txt")
        if not os.path.exists(self.dict_path):
            fail("ocr-dict-missing", "character dictionary not found: %s" % self.dict_path)
        # Индексы CTC: 0 — blank, затем строки словаря ровно в порядке файла
        # (замена пробелов в словаре не делаем), затем индекс пробела.
        self.chars = ["blank"]
        with open(self.dict_path, "r", encoding="utf-8") as handle:
            for line in handle:
                self.chars.append(line.rstrip("\n"))
        self.chars.append(" ")
        available = ort.get_available_providers()
        selected = [name for name in providers if name in available]
        if "CPUExecutionProvider" not in selected:
            selected.append("CPUExecutionProvider")
        self.det = ort.InferenceSession(self.det_path, providers=selected)
        self.rec = ort.InferenceSession(self.rec_path, providers=selected)
        used = self.det.get_providers()
        if require_gpu and not any(name.startswith(("CUDA", "Tensorrt")) for name in used):
            fail("cuda-ep-unavailable", "OCR models fell back to CPU; GPU is required", {"used": used})
        self.providers_used = used
        self.det_input = self.det.get_inputs()[0].name
        self.rec_input = self.rec.get_inputs()[0].name

    def detect_boxes(self, image, limit):
        """Детектор PP-OCRv5 (PPLCNetV3): размеры, кратные 32, вход BGR 0..1 без дополнительной нормализации."""
        height, width = image.shape[:2]
        scale = min(1.0, float(limit) / max(height, width))
        resized_h = max(32, int(round(height * scale / 32.0)) * 32)
        resized_w = max(32, int(round(width * scale / 32.0)) * 32)
        resized = cv2.resize(image, (resized_w, resized_h))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        output = self.det.run(None, {self.det_input: blob})[0]
        probability = np.asarray(output).reshape(resized_h, resized_w)
        mask = np.where(probability > 0.3, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 12:
                continue
            rect = cv2.minAreaRect(contour)
            (cx, cy), (bw, bh), angle = rect
            if min(bw, bh) < 3:
                continue
            if bw < bh:
                bw, bh = bh, bw
            expanded_w = bw * 1.4
            expanded_h = bh * 1.6
            points = cv2.boxPoints(((cx, cy), (expanded_w, expanded_h), angle))
            points[:, 0] = np.clip(points[:, 0] / scale, 0, width)
            points[:, 1] = np.clip(points[:, 1] / scale, 0, height)
            boxes.append(points.astype(np.float32))
        boxes.sort(key=lambda points: (float(np.min(points[:, 1])), float(np.min(points[:, 0]))))
        return boxes

    def recognize(self, image, box):
        """Распознаёт строку: кроп из bbox, вертикальная ориентация, вход 3x48xW."""
        points = np.array(box, dtype=np.float32)
        aspect = cv2.minAreaRect(points)[1]
        if aspect[0] < aspect[1]:
            # Вертикальный текст: поворачиваем кроп на 90 градусов по часовой стрелке.
            points = np.array([points[1], points[2], points[3], points[0]], dtype=np.float32)
        width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        if height < 4 or width < 4:
            return None
        ratio = width / float(height)
        target_width = max(8, min(320, int(48 * ratio)))
        target_height = max(8, min(160, int(320 / max(ratio, 0.1))))
        target = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(points, target)
        crop = cv2.warpPerspective(image, matrix, (target_width, target_height), borderValue=(255, 255, 255))
        if crop.shape[1] < crop.shape[0]:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        new_height = crop.shape[1] * 48 / max(1, crop.shape[0])
        resized = cv2.resize(crop, (int(min(320, new_height)), 48))
        padded = np.full((48, 320, 3), 255, dtype=np.uint8)
        limit = min(320, resized.shape[1])
        padded[:, :limit] = resized[:, :limit]
        blob = padded.astype(np.float32) / 255.0
        blob = ((blob - 0.5) / 0.5).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        output = self.rec.run(None, {self.rec_input: blob})[0]
        logits = np.asarray(output)
        per_step = logits.argmax(axis=-1).reshape(-1)
        probabilities = logits.reshape(-1, logits.shape[-1])
        texts = []
        scores = []
        previous = -1
        for step, index in enumerate(per_step):
            if index == previous:
                continue
            previous = int(index)
            if index == 0 or index >= len(self.chars):
                continue
            texts.append(self.chars[int(index)])
            scores.append(float(probabilities[step][index]))
        text = "".join(texts).strip()
        if not text or (scores and max(scores) < 0.3):
            return None
        score = float(np.mean(scores)) if scores else 0.0
        return {"text": text, "score": round(score, 4), "box": [[float(x), float(y)] for x, y in np.array(box, dtype=np.float32)]}

    def run_image(self, image, limit):
        boxes = self.detect_boxes(image, limit)
        lines = []
        for box in boxes:
            try:
                line = self.recognize(image, box)
            except Exception:
                line = None
            if line:
                lines.append(line)
        return merge_lines(lines)


def merge_lines(lines):
    """Склеивает соседние по высоте фрагменты строки (детектор часто режет цифры отдельно)."""
    def top(line):
        return min(point[1] for point in line["box"])

    def left(line):
        return min(point[0] for point in line["box"])

    ordered = sorted(lines, key=lambda line: (top(line), left(line)))
    merged = []
    for line in ordered:
        height = max(abs(p[1] - q[1]) for p in line["box"] for q in line["box"]) + 1
        placed = False
        for group in merged:
            gtop = min(point[1] for point in group[0]["box"])
            gbottom = max(point[1] for point in group[0]["box"])
            if not (top(line) > gbottom or gtop > max(point[1] for point in line["box"])):
                continue
            gleft = min(point[0] for point in group[0]["box"])
            gright = max(point[0] for point in group[0]["box"])
            gap = left(line) - gright if left(line) > gright else gleft - max(point[0] for point in line["box"])
            if gap <= height * 1.2:
                group.append(line)
                placed = True
                break
        if not placed:
            merged.append([line])
    result = []
    for group in merged:
        group.sort(key=left)
        text = "".join(item["text"] for item in group)
        result.append({
            "text": text,
            "score": round(sum(item["score"] for item in group) / len(group), 4),
            "box": group[0]["box"],
        })
    result.sort(key=top)
    return result


def read_video_frames(command, width, height):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = width * height * 3
    try:
        while True:
            buffer = process.stdout.read(frame_bytes)
            if not buffer or len(buffer) < frame_bytes:
                break
            yield np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3))
    finally:
        process.stdout.close()
        process.stderr.close()
        process.wait(timeout=10)


def extract_frame(ffmpeg, source, at_sec):
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(at_sec), "-i", source, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if not result.stdout:
        return None
    return cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config)

    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:
        fail("onnx-deps-missing", "onnxruntime is not installed: %s" % exc)

    model_dir = cfg.get("modelDir")
    if not model_dir:
        fail("invalid-args", "modelDir is required")
    device = str(cfg.get("device") or "cuda:0")
    require_gpu = device != "cpu"
    providers = cfg.get("providers") or ["CUDAExecutionProvider", "CPUExecutionProvider"]
    engine = PpOcr(model_dir, cfg.get("dictPath"), require_gpu, providers)
    emit({"type": "log", "message": "providers: %s" % ", ".join(engine.providers_used)})

    output_dir = cfg.get("outputDir") or os.path.join(os.getcwd(), "ocr-out")
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "ocr.jsonl")
    limit = int(cfg.get("detLimit") or 960)
    region = cfg.get("region")
    images = cfg.get("images") or []
    if cfg.get("source"):
        images = [cfg["source"]]
    if not images:
        fail("invalid-args", "source or images is required")

    started = time.time()
    samples = 0
    lines_total = 0
    last_emit = 0.0
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for image_path in images:
            if not os.path.exists(image_path):
                fail("input-missing", "file not found: %s" % image_path)
            video = image_path.lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".avi", ".ts"))
            if not video:
                image = cv2.imread(image_path)
                if image is None:
                    fail("input-missing", "cannot read image: %s" % image_path)
                samples += 1
                if region:
                    x, y, w, h = [int(region.get(key) or 0) for key in ("x", "y", "w", "h")]
                    image = image[y : y + h, x : x + w]
                lines = engine.run_image(image, limit)
                lines_total += len(lines)
                handle.write(json.dumps({"file": image_path, "lines": lines}, ensure_ascii=False) + "\n")
                continue

            ffmpeg = cfg.get("ffmpegPath") or "ffmpeg"
            at_sec = cfg.get("atSec") or []
            if at_sec:
                total = len(at_sec)
                for index, moment in enumerate(at_sec):
                    frame = extract_frame(ffmpeg, image_path, moment)
                    if frame is None:
                        continue
                    if region:
                        x, y, w, h = [int(region.get(key) or 0) for key in ("x", "y", "w", "h")]
                        frame = frame[y : y + h, x : x + w]
                    samples += 1
                    lines = engine.run_image(frame, limit)
                    lines_total += len(lines)
                    handle.write(json.dumps({"time": float(moment), "lines": lines}, ensure_ascii=False) + "\n")
                    percent = (index + 1) * 100.0 / max(1, total)
                    emit({"type": "progress", "phase": "ocr", "percent": percent, "frame": index + 1, "totalFrames": total})
                continue

            width = int(cfg.get("width") or 1280)
            height = int(cfg.get("height") or 720)
            fps = float(cfg.get("fps") or 1.0)
            duration = cfg.get("durationSec")
            total = max(1, int(float(duration) * fps)) if duration else None
            command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
            if cfg.get("startSec"):
                command += ["-ss", str(cfg["startSec"])]
            command += ["-i", image_path]
            if duration:
                command += ["-t", str(duration)]
            command += ["-vf", "fps=%s" % fps, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
            for frame in read_video_frames(command, width, height):
                if region:
                    x, y, w, h = [int(region.get(key) or 0) for key in ("x", "y", "w", "h")]
                    frame = frame[y : y + h, x : x + w]
                samples += 1
                lines = engine.run_image(frame, limit)
                lines_total += len(lines)
                handle.write(json.dumps({"time": round(samples / fps, 3), "lines": lines}, ensure_ascii=False) + "\n")
                now = time.time()
                if now - last_emit >= 0.4:
                    last_emit = now
                    rate = samples / max(1e-6, now - started)
                    percent = max(0.0, min(100.0, samples * 100.0 / total)) if total else None
                    eta = int(max(0.0, ((total or 0) - samples) / max(rate, 1e-6)) * 1000) if total else None
                    emit(
                        {
                            "type": "progress",
                            "phase": "ocr",
                            "percent": percent,
                            "frame": samples,
                            "totalFrames": total,
                            "fps": round(rate, 2),
                            "rate": round(rate, 3),
                            "etaMs": eta,
                        }
                    )

    emit(
        {
            "type": "result",
            "value": {
                "samples": samples,
                "lines": lines_total,
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
        fail("ocr-failed", str(exc), {"traceback": traceback.format_exc()[-4000:]})
