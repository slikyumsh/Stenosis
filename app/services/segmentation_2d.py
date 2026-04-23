from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects, skeletonize


def segment_vessels(image_path: Path) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    normalized = enhanced.astype(np.float32) / 255.0
    vesselness = frangi(normalized, sigmas=range(1, 4), alpha=0.5, beta=0.5, gamma=15)
    threshold = threshold_otsu(vesselness) if np.any(vesselness > 0) else 0.0
    mask = vesselness > threshold
    mask = binary_closing(mask, footprint=disk(2))
    mask = remove_small_objects(mask, min_size=64)
    mask = remove_small_holes(mask, area_threshold=64)

    return {
        "mask": mask.astype(np.uint8),
        "gray": image,
    }


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    cv2.imwrite(str(output_path), (mask.astype(np.uint8) * 255))


def save_mask_render(mask: np.ndarray, output_path: Path) -> None:
    rendered = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rendered[mask > 0] = (40, 220, 80)
    cv2.imwrite(str(output_path), rendered)


def save_segmentation_overlay(gray: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgb[mask > 0] = (40, 220, 80)
    cv2.imwrite(str(output_path), rgb)


def global_vessel_stats(mask: np.ndarray, pixel_spacing_mm: float) -> dict:
    if mask.max() == 0:
        return {
            "mask_area_px": 0,
            "mask_area_mm2": 0.0,
            "mean_diameter_mm": 0.0,
            "max_diameter_mm": 0.0,
        }

    edt = distance_transform_edt(mask.astype(bool)) * float(pixel_spacing_mm)
    skeleton = skeletonize(mask.astype(bool))
    diameters = edt[skeleton] * 2.0
    if diameters.size == 0:
        diameters = np.array([0.0], dtype=np.float32)

    return {
        "mask_area_px": int(mask.sum()),
        "mask_area_mm2": round(float(mask.sum() * pixel_spacing_mm * pixel_spacing_mm), 2),
        "mean_diameter_mm": round(float(diameters.mean()), 2),
        "max_diameter_mm": round(float(diameters.max()), 2),
    }
