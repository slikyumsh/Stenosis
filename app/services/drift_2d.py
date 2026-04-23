from __future__ import annotations

import glob
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from app.config import get_settings
from app.services.detection_2d import run_detection
from app.services.segmentation_2d import segment_vessels


FEATURE_NAMES = (
    "mean_intensity",
    "std_intensity",
    "entropy",
    "vessel_coverage_ratio",
    "detection_count",
    "mean_confidence",
    "bbox_coverage_ratio",
)
HISTOGRAM_BINS = 32
EPSILON = 1e-6


def _cache_path() -> Path:
    return get_settings().storage_root / "drift" / "reference_2d.json"


def _round(value: float) -> float:
    return round(float(value), 4)


def _load_gray(image_path: Path) -> np.ndarray:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return gray


def _normalize_distribution(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).copy()
    array = np.maximum(array, EPSILON)
    total = float(array.sum())
    if total <= 0:
        return np.full_like(array, 1.0 / max(len(array), 1))
    return array / total


def _normalized_histogram(gray: np.ndarray) -> np.ndarray:
    histogram = cv2.calcHist([gray], [0], None, [HISTOGRAM_BINS], [0, 256]).ravel().astype(np.float64)
    return _normalize_distribution(histogram)


def _histogram_entropy(gray: np.ndarray) -> float:
    probabilities = _normalized_histogram(gray)
    return float(-(probabilities * np.log2(probabilities)).sum())


def _bbox_coverage_ratio(boxes: list[dict], width: int, height: int) -> float:
    image_area = max(float(width * height), 1.0)
    total_area = 0.0
    for box in boxes:
        box_width = max(float(box["x2"]) - float(box["x1"]), 0.0)
        box_height = max(float(box["y2"]) - float(box["y1"]), 0.0)
        total_area += box_width * box_height
    return total_area / image_area


def _extract_features(gray: np.ndarray, mask: np.ndarray, boxes: list[dict]) -> dict[str, float]:
    height, width = gray.shape[:2]
    confidences = [float(box["confidence"]) for box in boxes]
    return {
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
        "entropy": _histogram_entropy(gray),
        "vessel_coverage_ratio": float(mask.astype(bool).mean()),
        "detection_count": float(len(boxes)),
        "mean_confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
        "bbox_coverage_ratio": _bbox_coverage_ratio(boxes, width, height),
    }


def _summarize_feature(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": _round(array.mean()),
        "std": _round(array.std(ddof=0)),
    }


def _build_reference_profile() -> dict:
    settings = get_settings()
    image_paths = [
        Path(path)
        for path in sorted(glob.glob(settings.drift_reference_glob))[: settings.drift_reference_sample_limit]
    ]
    feature_buckets = {name: [] for name in FEATURE_NAMES}
    intensity_histograms: list[np.ndarray] = []
    used_paths: list[str] = []

    for image_path in image_paths:
        gray = _load_gray(image_path)
        segmentation = segment_vessels(image_path)
        detection = run_detection(image_path)
        features = _extract_features(gray, segmentation["mask"], detection["boxes"])
        for name in FEATURE_NAMES:
            feature_buckets[name].append(features[name])
        intensity_histograms.append(_normalized_histogram(gray))
        used_paths.append(str(image_path))

    reference_histogram = np.mean(intensity_histograms, axis=0) if intensity_histograms else np.zeros(HISTOGRAM_BINS)
    reference_histogram = _normalize_distribution(reference_histogram)

    profile = {
        "sample_count": len(used_paths),
        "reference_glob": settings.drift_reference_glob,
        "feature_stats": {
            name: _summarize_feature(values)
            for name, values in feature_buckets.items()
        },
        "intensity_histogram_mean": [_round(value) for value in reference_histogram.tolist()],
        "intensity_histogram_bins": HISTOGRAM_BINS,
        "reference_examples": used_paths[:5],
    }

    cache_path = _cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, ensure_ascii=True, indent=2)
    return profile


def _profile_is_valid(profile: dict) -> bool:
    return (
        isinstance(profile, dict)
        and "feature_stats" in profile
        and "intensity_histogram_mean" in profile
        and profile.get("intensity_histogram_bins") == HISTOGRAM_BINS
    )


@lru_cache(maxsize=1)
def get_reference_profile() -> dict:
    cache_path = _cache_path()
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            profile = json.load(handle)
        if _profile_is_valid(profile):
            return profile
    return _build_reference_profile()


def _normalized_score(value: float, mean: float, std: float) -> float:
    baseline = max(float(std), max(abs(float(mean)) * 0.1, 1e-3))
    z_value = abs(float(value) - float(mean)) / baseline
    return min(z_value / 3.0, 1.0)


def _level_from_score(score: float) -> str:
    if score >= 0.65:
        return "critical"
    if score >= 0.35:
        return "warning"
    return "normal"


def _level_from_thresholds(value: float, warning_threshold: float, critical_threshold: float) -> str:
    if value >= critical_threshold:
        return "critical"
    if value >= warning_threshold:
        return "warning"
    return "normal"


def _max_level(levels: list[str]) -> str:
    priorities = {"unknown": 0, "normal": 1, "warning": 2, "critical": 3}
    return max(levels, key=lambda level: priorities.get(level, 0))


def _population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    expected_safe = _normalize_distribution(expected)
    actual_safe = _normalize_distribution(actual)
    return float(np.sum((actual_safe - expected_safe) * np.log(actual_safe / expected_safe)))


def _jensen_shannon_divergence(expected: np.ndarray, actual: np.ndarray) -> float:
    distance = float(jensenshannon(_normalize_distribution(expected), _normalize_distribution(actual), base=2.0))
    return distance * distance


def _wasserstein_from_histograms(expected: np.ndarray, actual: np.ndarray) -> float:
    bin_edges = np.linspace(0.0, 256.0, len(expected) + 1, dtype=np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return float(
        wasserstein_distance(
            bin_centers,
            bin_centers,
            u_weights=_normalize_distribution(expected),
            v_weights=_normalize_distribution(actual),
        )
    )


def _build_academic_metrics(gray: np.ndarray, reference: dict) -> dict:
    current_histogram = _normalized_histogram(gray)
    reference_histogram = _normalize_distribution(np.asarray(reference.get("intensity_histogram_mean", []), dtype=np.float64))

    psi_value = _population_stability_index(reference_histogram, current_histogram)
    js_value = _jensen_shannon_divergence(reference_histogram, current_histogram)
    wasserstein_value = _wasserstein_from_histograms(reference_histogram, current_histogram)

    psi_level = _level_from_thresholds(psi_value, warning_threshold=0.1, critical_threshold=0.25)
    js_level = _level_from_thresholds(js_value, warning_threshold=0.02, critical_threshold=0.05)
    wasserstein_level = _level_from_thresholds(wasserstein_value, warning_threshold=6.0, critical_threshold=12.0)

    academic_score = float(
        (
            min(psi_value / 0.25, 1.0)
            + min(js_value / 0.05, 1.0)
            + min(wasserstein_value / 12.0, 1.0)
        )
        / 3.0
    )

    return {
        "score": _round(academic_score),
        "level": _max_level([psi_level, js_level, wasserstein_level]),
        "metrics": {
            "intensity_distribution": {
                "description": "Distributional drift on grayscale histogram.",
                "bins": HISTOGRAM_BINS,
                "population_stability_index": {
                    "value": _round(psi_value),
                    "warning_gte": 0.1,
                    "critical_gte": 0.25,
                    "level": psi_level,
                },
                "jensen_shannon_divergence": {
                    "value": _round(js_value),
                    "warning_gte": 0.02,
                    "critical_gte": 0.05,
                    "level": js_level,
                },
                "wasserstein_distance": {
                    "value": _round(wasserstein_value),
                    "warning_gte": 6.0,
                    "critical_gte": 12.0,
                    "level": wasserstein_level,
                    "unit": "intensity_levels",
                },
            }
        },
    }


def analyze_request_drift(image_path: Path, mask: np.ndarray, boxes: list[dict]) -> dict:
    gray = _load_gray(image_path)
    current = _extract_features(gray, mask, boxes)
    reference = get_reference_profile()
    feature_stats = reference.get("feature_stats", {})

    if not feature_stats or reference.get("sample_count", 0) == 0:
        return {
            "reference_sample_count": 0,
            "custom_score": 0.0,
            "academic_score": 0.0,
            "combined_score": 0.0,
            "custom_level": "unknown",
            "academic_level": "unknown",
            "level": "unknown",
            "methods": {},
            "academic_metrics": {},
            "current_features": {name: _round(value) for name, value in current.items()},
        }

    custom_methods = {
        "brightness_shift": ("mean_intensity", "Mean grayscale intensity shift."),
        "contrast_shift": ("std_intensity", "Intensity spread shift."),
        "texture_entropy_shift": ("entropy", "Histogram entropy shift."),
        "vessel_coverage_shift": ("vessel_coverage_ratio", "Segmented vessel coverage shift."),
        "detection_density_shift": ("bbox_coverage_ratio", "Detected bbox area ratio shift."),
        "detection_confidence_shift": ("mean_confidence", "Mean detector confidence shift."),
    }

    methods_payload = {}
    custom_scores: list[float] = []
    for method_name, (feature_name, description) in custom_methods.items():
        stats = feature_stats.get(feature_name, {"mean": 0.0, "std": 0.0})
        score = _normalized_score(current[feature_name], stats["mean"], stats["std"])
        methods_payload[method_name] = {
            "description": description,
            "feature_name": feature_name,
            "current_value": _round(current[feature_name]),
            "reference_mean": _round(stats["mean"]),
            "reference_std": _round(stats["std"]),
            "score": _round(score),
        }
        custom_scores.append(score)

    custom_score = float(sum(custom_scores) / max(len(custom_scores), 1))
    custom_level = _level_from_score(custom_score)
    academic = _build_academic_metrics(gray, reference)
    combined_score = float((custom_score + float(academic["score"])) / 2.0)
    combined_level = _max_level([custom_level, academic["level"]])

    return {
        "reference_sample_count": int(reference.get("sample_count", 0)),
        "custom_score": _round(custom_score),
        "academic_score": _round(academic["score"]),
        "combined_score": _round(combined_score),
        "custom_level": custom_level,
        "academic_level": academic["level"],
        "level": combined_level,
        "thresholds": {
            "custom_normal_lt": 0.35,
            "custom_warning_lt": 0.65,
            "custom_critical_gte": 0.65,
        },
        "methods": methods_payload,
        "academic_metrics": academic["metrics"],
        "current_features": {name: _round(value) for name, value in current.items()},
        "reference_examples": reference.get("reference_examples", []),
    }
