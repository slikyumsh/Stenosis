from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("APP_YOLO_MODEL_PATH", str(ROOT / "best.onnx"))

from app.services.detection_2d import get_yolo_model, run_detection  # noqa: E402
from app.services.segmentation_2d import segment_vessels  # noqa: E402
from tools.eval_2d_detection_bootstrap import ImageRecord, evaluate, yolo_to_xyxy  # noqa: E402


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def detection_dict_to_arrays(boxes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    xyxy = np.asarray([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes], dtype=np.float32)
    scores = np.asarray([b["confidence"] for b in boxes], dtype=np.float32)
    return xyxy, scores


def find_ground_truth_mask(image_path: Path) -> Path | None:
    candidates = [
        image_path.with_name(f"{image_path.stem}_mask.png"),
        image_path.with_name(f"{image_path.stem}_mask.bmp"),
        image_path.with_name(f"{image_path.stem}_seg.png"),
        image_path.with_name(f"{image_path.stem}_seg.bmp"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-size", type=int, default=10)
    parser.add_argument("--test-dir", type=Path, default=ROOT / "data" / "test")
    parser.add_argument("--output", type=Path, default=ROOT / "temp_2d_subset_metrics_10cases.json")
    parser.add_argument("--timing-conf", type=float, default=0.25)
    parser.add_argument("--eval-conf", type=float, default=0.001)
    args = parser.parse_args()

    image_paths = sorted(args.test_dir.glob("*.bmp"))[: args.subset_size]
    if not image_paths:
        raise SystemExit(f"No BMP files found in {args.test_dir}")

    # Warm up the detector once to avoid counting model initialization.
    get_yolo_model()
    _ = run_detection(image_paths[0], confidence_threshold=args.timing_conf)

    records: list[ImageRecord] = []
    per_image: list[dict[str, object]] = []
    det_times: list[float] = []
    seg_times: list[float] = []
    combo_times: list[float] = []
    gt_masks_found = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        height, width = image.shape[:2]
        gt_boxes = yolo_to_xyxy(image_path.with_suffix(".txt"), width, height)

        detection_for_eval = run_detection(image_path, confidence_threshold=args.eval_conf)
        pred_boxes, pred_scores = detection_dict_to_arrays(detection_for_eval["boxes"])
        records.append(
            ImageRecord(
                image_path=image_path,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
            )
        )

        t0 = time.perf_counter()
        detection_for_timing = run_detection(image_path, confidence_threshold=args.timing_conf)
        det_elapsed = time.perf_counter() - t0

        t1 = time.perf_counter()
        segmentation = segment_vessels(image_path)
        seg_elapsed = time.perf_counter() - t1

        gt_mask = find_ground_truth_mask(image_path)
        if gt_mask is not None:
            gt_masks_found += 1

        det_times.append(det_elapsed)
        seg_times.append(seg_elapsed)
        combo_times.append(det_elapsed + seg_elapsed)
        per_image.append(
            {
                "image": image_path.name,
                "gt_box_count": int(len(gt_boxes)),
                "pred_box_count_eval_conf": int(len(pred_boxes)),
                "pred_box_count_timing_conf": int(len(detection_for_timing["boxes"])),
                "detection_seconds": float(det_elapsed),
                "segmentation_seconds": float(seg_elapsed),
                "combined_seconds": float(det_elapsed + seg_elapsed),
                "segmentation_mask_area_px": int(segmentation["mask"].sum()),
            }
        )

    payload = {
        "subset_policy": "first_sorted_test_images",
        "subset_size": len(image_paths),
        "images": [path.name for path in image_paths],
        "model_path": str(ROOT / "best.onnx"),
        "detection_eval_confidence_threshold": args.eval_conf,
        "detection_timing_confidence_threshold": args.timing_conf,
        "detection_metrics": evaluate(records),
        "timing_seconds": {
            "detection_only": summarize(det_times),
            "segmentation_only": summarize(seg_times),
            "detector_plus_segmentor": summarize(combo_times),
        },
        "segmentation_quality": {
            "ground_truth_masks_found": gt_masks_found,
            "quality_metrics_available": gt_masks_found == len(image_paths),
            "note": (
                "Ground-truth 2D masks were not found for this subset, so Dice/IoU were not computed."
                if gt_masks_found < len(image_paths)
                else "Ground-truth masks were found for the whole subset."
            ),
        },
        "per_image": per_image,
    }

    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
