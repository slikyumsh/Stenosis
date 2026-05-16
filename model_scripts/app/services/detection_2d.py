from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import cv2
from ultralytics import YOLO

from app.config import get_settings


@lru_cache(maxsize=1)
def get_yolo_model() -> YOLO:
    return YOLO(str(get_settings().yolo_model_path))


def run_detection(image_path: Path, confidence_threshold: float = 0.25) -> dict:
    model = get_yolo_model()
    start = time.perf_counter()
    results = model.predict(source=str(image_path), conf=confidence_threshold, verbose=False)
    elapsed = time.perf_counter() - start
    result = results[0]

    boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            boxes.append(
                {
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                    "confidence": float(box.conf[0].item()),
                    "class_id": int(box.cls[0].item()),
                }
            )

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    height, width = image.shape[:2]
    return {
        "boxes": boxes,
        "width": width,
        "height": height,
        "inference_seconds": elapsed,
        "image": image,
    }


def save_detection_overlay(image, boxes: list[dict], output_path: Path) -> None:
    overlay = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(box[key]) for key in ("x1", "y1", "x2", "y2")]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 70, 230), 2)
        label = f"{box['confidence']:.2f}"
        cv2.putText(
            overlay,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 70, 230),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), overlay)
