from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def _round_metric(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _stenosis_percentages(min_diameter_mm: float, mean_diameter_mm: float) -> dict:
    if mean_diameter_mm <= 0:
        return {
            "stenosis_ratio_percent": 0.0,
            "stenosis_narrowing_percent": 0.0,
        }
    ratio = (min_diameter_mm / mean_diameter_mm) * 100.0
    ratio = max(0.0, min(100.0, ratio))
    return {
        "stenosis_ratio_percent": _round_metric(ratio),
        "stenosis_narrowing_percent": _round_metric(100.0 - ratio),
    }


def bbox_measurements(boxes: list[dict], pixel_spacing_mm: float) -> list[dict]:
    measurements = []
    for box in boxes:
        width_px = max(0.0, box["x2"] - box["x1"])
        height_px = max(0.0, box["y2"] - box["y1"])
        width_mm = width_px * pixel_spacing_mm
        height_mm = height_px * pixel_spacing_mm
        measurements.append(
            {
                **box,
                "width_px": _round_metric(width_px),
                "height_px": _round_metric(height_px),
                "width_mm": _round_metric(width_mm),
                "height_mm": _round_metric(height_mm),
                "area_mm2": _round_metric(width_mm * height_mm),
                "estimated_stenosis_size_mm": _round_metric(max(width_mm, height_mm)),
            }
        )
    return measurements


def local_vessel_stats(mask: np.ndarray, box: dict, pixel_spacing_mm: float) -> dict:
    x1 = max(0, int(box["x1"]))
    y1 = max(0, int(box["y1"]))
    x2 = max(x1 + 1, int(box["x2"]))
    y2 = max(y1 + 1, int(box["y2"]))
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0 or roi.max() == 0:
        return {
            "local_vessel_area_px": 0,
            "local_min_diameter_mm": 0.0,
            "local_mean_diameter_mm": 0.0,
            "local_max_diameter_mm": 0.0,
            "stenosis_ratio_percent": 0.0,
            "stenosis_narrowing_percent": 0.0,
        }

    edt = distance_transform_edt(roi.astype(bool)) * float(pixel_spacing_mm)
    skeleton = skeletonize(roi.astype(bool))
    diameters = edt[skeleton] * 2.0
    if diameters.size == 0:
        diameters = np.array([0.0], dtype=np.float32)

    min_diameter = float(diameters.min())
    mean_diameter = float(diameters.mean())
    max_diameter = float(diameters.max())
    return {
        "local_vessel_area_px": int(roi.sum()),
        "local_min_diameter_mm": _round_metric(min_diameter),
        "local_mean_diameter_mm": _round_metric(mean_diameter),
        "local_max_diameter_mm": _round_metric(max_diameter),
        **_stenosis_percentages(min_diameter, mean_diameter),
    }
