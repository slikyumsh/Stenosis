from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "test"
MODEL_PATH = ROOT / "YOLO_Stenosis_Detection" / "YOLOv8m_training5" / "weights" / "best.pt"
OUT_JSON = ROOT / "section_2_1_detection_eval.json"
OUT_IMAGE = ROOT / "section_2_1_detection_example.png"


IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


@dataclass
class ImageRecord:
    image_path: Path
    gt_boxes: np.ndarray
    pred_boxes: np.ndarray
    pred_scores: np.ndarray


def yolo_to_xyxy(label_path: Path, width: int, height: int) -> np.ndarray:
    boxes: list[list[float]] = []
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = map(float, parts)
        x1 = (xc - w / 2.0) * width
        y1 = (yc - h / 2.0) * height
        x2 = (xc + w / 2.0) * width
        y2 = (yc + h / 2.0) * height
        boxes.append([x1, y1, x2, y2])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = np.maximum(area1 + area2 - inter, 1e-9)
    return inter / union


def compute_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(records: list[ImageRecord], iou_thresholds: np.ndarray = IOU_THRESHOLDS) -> dict[str, float]:
    n_gt = int(sum(len(r.gt_boxes) for r in records))
    if n_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0}

    aps: list[float] = []
    precision_at_50 = 0.0
    recall_at_50 = 0.0

    for thr in iou_thresholds:
        confs: list[float] = []
        tps: list[int] = []
        fps: list[int] = []

        for record in records:
            gt = record.gt_boxes
            preds = record.pred_boxes
            scores = record.pred_scores
            used = np.zeros(len(gt), dtype=bool)
            order = np.argsort(-scores)

            for idx in order:
                box = preds[idx]
                conf = float(scores[idx])
                if len(gt) == 0:
                    confs.append(conf)
                    tps.append(0)
                    fps.append(1)
                    continue

                ious = box_iou(box, gt)
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])
                if best_iou >= thr and not used[best_idx]:
                    used[best_idx] = True
                    confs.append(conf)
                    tps.append(1)
                    fps.append(0)
                else:
                    confs.append(conf)
                    tps.append(0)
                    fps.append(1)

        if not confs:
            aps.append(0.0)
            continue

        order = np.argsort(-np.asarray(confs, dtype=np.float32))
        tp_cum = np.cumsum(np.asarray(tps, dtype=np.float32)[order])
        fp_cum = np.cumsum(np.asarray(fps, dtype=np.float32)[order])
        rec = tp_cum / max(n_gt, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap = compute_ap(rec, prec)
        aps.append(ap)

        if abs(thr - 0.5) < 1e-9:
            f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
            best = int(np.argmax(f1)) if len(f1) else 0
            precision_at_50 = float(prec[best]) if len(prec) else 0.0
            recall_at_50 = float(rec[best]) if len(rec) else 0.0

    return {
        "precision": precision_at_50,
        "recall": recall_at_50,
        "map50": float(aps[0]) if aps else 0.0,
        "map50_95": float(np.mean(aps)) if aps else 0.0,
    }


def bootstrap(records: list[ImageRecord], n_boot: int = 2000, seed: int = 42) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(records)
    samples = {k: [] for k in ("precision", "recall", "map50", "map50_95")}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_records = [records[i] for i in idx]
        metrics = evaluate(sample_records)
        for key, value in metrics.items():
            samples[key].append(value)

    out: dict[str, dict[str, float]] = {}
    for key, values in samples.items():
        arr = np.asarray(values, dtype=np.float64)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[key] = {
            "ci95_low": float(lo),
            "ci95_high": float(hi),
        }
    return out


def draw_example(image_path: Path, gt_boxes: np.ndarray, pred_boxes: np.ndarray, pred_scores: np.ndarray) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    order = np.argsort(-pred_scores)
    for idx in order[:3]:
        x1, y1, x2, y2 = map(int, pred_boxes[idx])
        score = float(pred_scores[idx])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            f"pred {score:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        "green: ground truth | red: prediction",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(OUT_IMAGE), image)


def main() -> None:
    image_paths = sorted(TEST_DIR.glob("*.bmp"))
    model = YOLO(str(MODEL_PATH))

    records: list[ImageRecord] = []
    example_saved = False

    for result in model.predict(
        source=[str(p) for p in image_paths],
        imgsz=800,
        conf=0.001,
        iou=0.7,
        device="cpu",
        stream=True,
        verbose=False,
    ):
        image_path = Path(result.path)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        label_path = image_path.with_suffix(".txt")
        gt_boxes = yolo_to_xyxy(label_path, w, h)

        if result.boxes is None or len(result.boxes) == 0:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)
        else:
            pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_scores = result.boxes.conf.cpu().numpy().astype(np.float32)

        records.append(
            ImageRecord(
                image_path=image_path,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
            )
        )

        if not example_saved and len(gt_boxes) > 0 and len(pred_boxes) > 0:
            draw_example(image_path, gt_boxes, pred_boxes, pred_scores)
            example_saved = True

    point = evaluate(records)
    ci = bootstrap(records)

    payload = {
        "model_path": str(MODEL_PATH),
        "split": "test",
        "num_images": len(records),
        "metrics": {
            key: {
                "value": float(point[key]),
                **ci[key],
            }
            for key in ("precision", "recall", "map50", "map50_95")
        },
        "example_image": str(OUT_IMAGE),
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
