from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
STENOSIS3D_ROOT = (ROOT / ".." / "Stenosis3D").resolve()
STENOSIS3D_SCRIPTS = STENOSIS3D_ROOT / "scripts"
if str(STENOSIS3D_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STENOSIS3D_SCRIPTS))

from render_case_3d import (  # type: ignore  # noqa: E402
    edt_radius_mm,
    longest_path_on_skeleton_graph,
    skeleton_points_and_edges_26,
)
from simulate_stenosis_and_detect import make_three_projections, simulate_stenosis_pinch  # type: ignore  # noqa: E402
from stenosis_proj_detect import load_mask_nii, run_yolo_on_image, to_3ch  # type: ignore  # noqa: E402


@dataclass
class MethodResult:
    estimated_diameter_mm: float | None
    elapsed_seconds: float
    abs_diameter_error_mm: float | None
    abs_stenosis_error_pp: float | None
    success: bool
    details: dict


@dataclass
class CaseResult:
    case_id: str
    shape_xyz: list[int]
    target_diameter_mm: float
    target_stenosis_percent: float
    r0_mm: float
    ratio: float
    half_length_mm: float
    method_1_simple_3d_ball: MethodResult
    method_2_proj_cube_3d_ball: MethodResult
    method_3_proj_2d_circles_reconstruct: MethodResult


@dataclass
class YoloBundle:
    model: YOLO
    device: str
    conf: float
    iou: float


def positive_min(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    finite = finite[finite > 0]
    if finite.size == 0:
        return None
    return float(finite.min())


def estimate_min_diameter_3d(mask_u8: np.ndarray, pixdim: tuple[float, float, float]) -> tuple[float | None, dict]:
    skel = skeletonize(mask_u8.astype(bool), method="lee").astype(np.uint8)
    pts, edges = skeleton_points_and_edges_26(skel)
    if pts.size == 0:
        return None, {"reason": "empty_skeleton"}

    main_path = longest_path_on_skeleton_graph(pts, edges)
    if main_path.size == 0:
        main_path = pts[:1].astype(np.int32)

    radius_map = edt_radius_mm(mask_u8, pixdim)
    rr = radius_map[main_path[:, 0], main_path[:, 1], main_path[:, 2]]
    r_min = positive_min(rr)
    if r_min is None:
        return None, {"reason": "no_positive_radius", "centerline_points": int(main_path.shape[0])}

    return 2.0 * r_min, {"centerline_points": int(main_path.shape[0])}


def min_circle_diameter_on_projection(
    projection_mask_u8: np.ndarray,
    spacing_2d: tuple[float, float],
) -> tuple[float | None, dict]:
    skel = skeletonize(projection_mask_u8.astype(bool)).astype(np.uint8)
    if skel.max() == 0:
        return None, {"reason": "empty_2d_skeleton"}

    radius_map = distance_transform_edt(projection_mask_u8.astype(bool), sampling=spacing_2d).astype(np.float32)
    rr = radius_map[skel > 0]
    r_min = positive_min(rr)
    if r_min is None:
        return None, {"reason": "no_positive_2d_radius", "skeleton_points": int(skel.sum())}
    return 2.0 * r_min, {"skeleton_points": int(skel.sum())}


def clamp_range(lo: int, hi: int, size: int) -> tuple[int, int]:
    lo = max(0, min(lo, size))
    hi = max(0, min(hi, size))
    if hi <= lo:
        hi = min(size, lo + 1)
    return lo, hi


def bbox_xyxy_to_ranges(box_xyxy: list[float], shape_2d: tuple[int, int], margin: int = 2) -> tuple[int, int, int, int]:
    h, w = int(shape_2d[0]), int(shape_2d[1])
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    row0, row1 = clamp_range(y1 - margin, y2 + margin + 1, h)
    col0, col1 = clamp_range(x1 - margin, x2 + margin + 1, w)
    return row0, row1, col0, col1


def pick_best_detection(detections: list[dict]) -> dict | None:
    if not detections:
        return None
    return max(detections, key=lambda item: float(item.get("conf", 0.0)))


def detect_projections_timed(
    mask_u8: np.ndarray,
    yolo: YoloBundle,
) -> tuple[dict[str, dict | None], dict[str, list[dict]], float]:
    t0 = time.perf_counter()
    projections = make_three_projections(mask_u8)
    raw_detections: dict[str, list[dict]] = {}
    best_detections: dict[str, dict | None] = {}
    for view_name, proj_u8 in projections.items():
        dets = run_yolo_on_image(
            yolo.model,
            to_3ch(proj_u8),
            conf=yolo.conf,
            iou=yolo.iou,
            device=yolo.device,
        )
        raw_detections[view_name] = dets
        best_detections[view_name] = pick_best_detection(dets)
    elapsed = time.perf_counter() - t0
    return best_detections, raw_detections, elapsed


def build_cube_from_yolo_boxes(
    mask_u8: np.ndarray,
    best_boxes: dict[str, dict | None],
) -> tuple[tuple[int, int, int, int, int, int] | None, dict]:
    sx, sy, sz = mask_u8.shape
    ranges: dict[str, tuple[int, int, int, int]] = {}

    axial = best_boxes.get("axial")
    coronal = best_boxes.get("coronal")
    sagittal = best_boxes.get("sagittal")
    if axial is not None:
        ranges["axial"] = bbox_xyxy_to_ranges(axial["xyxy"], (sx, sy))
    if coronal is not None:
        ranges["coronal"] = bbox_xyxy_to_ranges(coronal["xyxy"], (sx, sz))
    if sagittal is not None:
        ranges["sagittal"] = bbox_xyxy_to_ranges(sagittal["xyxy"], (sy, sz))

    info = {
        "detected_views": sorted(ranges.keys()),
        "boxes_xyxy": {
            key: (best_boxes[key]["xyxy"] if best_boxes.get(key) is not None else None)
            for key in ("axial", "coronal", "sagittal")
        },
        "confidences": {
            key: (float(best_boxes[key]["conf"]) if best_boxes.get(key) is not None else None)
            for key in ("axial", "coronal", "sagittal")
        },
    }
    if len(ranges) < 2:
        return None, {**info, "reason": "fewer_than_two_detected_views"}

    x_candidates: list[tuple[int, int]] = []
    y_candidates: list[tuple[int, int]] = []
    z_candidates: list[tuple[int, int]] = []

    if "axial" in ranges:
        row0, row1, col0, col1 = ranges["axial"]
        x_candidates.append((row0, row1))
        y_candidates.append((col0, col1))
    if "coronal" in ranges:
        row0, row1, col0, col1 = ranges["coronal"]
        x_candidates.append((row0, row1))
        z_candidates.append((col0, col1))
    if "sagittal" in ranges:
        row0, row1, col0, col1 = ranges["sagittal"]
        y_candidates.append((row0, row1))
        z_candidates.append((col0, col1))

    if not x_candidates or not y_candidates or not z_candidates:
        return None, {**info, "reason": "insufficient_axis_constraints"}

    def intersect_or_union(candidates: list[tuple[int, int]], size: int) -> tuple[int, int]:
        lo = max(item[0] for item in candidates)
        hi = min(item[1] for item in candidates)
        if hi > lo:
            return clamp_range(lo, hi, size)
        lo = min(item[0] for item in candidates)
        hi = max(item[1] for item in candidates)
        return clamp_range(lo, hi, size)

    x0, x1 = intersect_or_union(x_candidates, sx)
    y0, y1 = intersect_or_union(y_candidates, sy)
    z0, z1 = intersect_or_union(z_candidates, sz)
    return (x0, x1, y0, y1, z0, z1), info


def method_simple_3d_ball(
    stenosed_mask_u8: np.ndarray,
    pixdim: tuple[float, float, float],
    target_diameter_mm: float,
    target_stenosis_percent: float,
    r0_mm: float,
) -> MethodResult:
    t0 = time.perf_counter()
    estimated_diameter_mm, details = estimate_min_diameter_3d(stenosed_mask_u8, pixdim)
    elapsed = time.perf_counter() - t0
    if estimated_diameter_mm is None:
        return MethodResult(None, elapsed, None, None, False, details)

    estimated_stenosis = max(0.0, (1.0 - (estimated_diameter_mm / (2.0 * r0_mm))) * 100.0) if r0_mm > 0 else 0.0
    return MethodResult(
        estimated_diameter_mm=estimated_diameter_mm,
        elapsed_seconds=elapsed,
        abs_diameter_error_mm=abs(estimated_diameter_mm - target_diameter_mm),
        abs_stenosis_error_pp=abs(estimated_stenosis - target_stenosis_percent),
        success=True,
        details=details,
    )


def method_proj_cube_3d_ball(
    stenosed_mask_u8: np.ndarray,
    pixdim: tuple[float, float, float],
    target_diameter_mm: float,
    target_stenosis_percent: float,
    r0_mm: float,
    best_boxes: dict[str, dict | None],
    raw_boxes: dict[str, list[dict]],
    yolo_elapsed_seconds: float,
) -> MethodResult:
    t0 = time.perf_counter()
    cube_bounds, cube_info = build_cube_from_yolo_boxes(stenosed_mask_u8, best_boxes)
    if cube_bounds is None:
        elapsed = yolo_elapsed_seconds + (time.perf_counter() - t0)
        return MethodResult(
            None,
            elapsed,
            None,
            None,
            False,
            {**cube_info, "raw_detections_per_view": {k: len(v) for k, v in raw_boxes.items()}},
        )

    x0, x1, y0, y1, z0, z1 = cube_bounds
    crop = stenosed_mask_u8[x0:x1, y0:y1, z0:z1]
    estimated_diameter_mm, details = estimate_min_diameter_3d(crop, pixdim)
    elapsed = yolo_elapsed_seconds + (time.perf_counter() - t0)
    details = {
        **cube_info,
        **details,
        "cube_bounds_xyz": cube_bounds,
        "raw_detections_per_view": {k: len(v) for k, v in raw_boxes.items()},
    }
    if estimated_diameter_mm is None:
        return MethodResult(None, elapsed, None, None, False, details)

    estimated_stenosis = max(0.0, (1.0 - (estimated_diameter_mm / (2.0 * r0_mm))) * 100.0) if r0_mm > 0 else 0.0
    return MethodResult(
        estimated_diameter_mm=estimated_diameter_mm,
        elapsed_seconds=elapsed,
        abs_diameter_error_mm=abs(estimated_diameter_mm - target_diameter_mm),
        abs_stenosis_error_pp=abs(estimated_stenosis - target_stenosis_percent),
        success=True,
        details=details,
    )


def method_proj_2d_circles_reconstruct(
    stenosed_mask_u8: np.ndarray,
    pixdim: tuple[float, float, float],
    target_diameter_mm: float,
    target_stenosis_percent: float,
    r0_mm: float,
    best_boxes: dict[str, dict | None],
    raw_boxes: dict[str, list[dict]],
    yolo_elapsed_seconds: float,
) -> MethodResult:
    t0 = time.perf_counter()
    projections = make_three_projections(stenosed_mask_u8)

    per_view_diameters: dict[str, float | None] = {}
    per_view_info: dict[str, dict] = {}
    view_setup = {
        "axial": ((pixdim[0], pixdim[1]), projections["axial"]),
        "coronal": ((pixdim[0], pixdim[2]), projections["coronal"]),
        "sagittal": ((pixdim[1], pixdim[2]), projections["sagittal"]),
    }
    for view_name, (spacing_2d, proj_u8) in view_setup.items():
        det = best_boxes.get(view_name)
        if det is None:
            per_view_diameters[view_name] = None
            per_view_info[view_name] = {"reason": "no_detection"}
            continue
        row0, row1, col0, col1 = bbox_xyxy_to_ranges(det["xyxy"], proj_u8.shape)
        crop = (proj_u8[row0:row1, col0:col1] > 0).astype(np.uint8)
        diam, info = min_circle_diameter_on_projection(crop, spacing_2d)
        per_view_diameters[view_name] = diam
        per_view_info[view_name] = {
            **info,
            "bbox_rows_cols": [row0, row1, col0, col1],
            "conf": float(det["conf"]),
        }

    elapsed = yolo_elapsed_seconds + (time.perf_counter() - t0)
    valid_diameters = [value for value in per_view_diameters.values() if value is not None]
    details = {
        "raw_detections_per_view": {k: len(v) for k, v in raw_boxes.items()},
        "view_details": per_view_info,
        "view_diameters_mm": per_view_diameters,
    }
    if len(valid_diameters) < 2:
        return MethodResult(None, elapsed, None, None, False, {**details, "reason": "fewer_than_two_views_with_diameter"})

    estimated_diameter_mm = float(min(valid_diameters))
    estimated_stenosis = max(0.0, (1.0 - (estimated_diameter_mm / (2.0 * r0_mm))) * 100.0) if r0_mm > 0 else 0.0
    return MethodResult(
        estimated_diameter_mm=estimated_diameter_mm,
        elapsed_seconds=elapsed,
        abs_diameter_error_mm=abs(estimated_diameter_mm - target_diameter_mm),
        abs_stenosis_error_pp=abs(estimated_stenosis - target_stenosis_percent),
        success=True,
        details=details,
    )


def summarize(values: list[float | None]) -> dict:
    numeric = [float(v) for v in values if v is not None]
    if not numeric:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    std = 0.0 if len(numeric) == 1 else statistics.pstdev(numeric)
    return {
        "n": len(numeric),
        "mean": float(statistics.mean(numeric)),
        "std": float(std),
        "min": float(min(numeric)),
        "max": float(max(numeric)),
    }


def build_summary(case_results: list[CaseResult]) -> dict:
    methods = {
        "method_1_simple_3d_ball": [c.method_1_simple_3d_ball for c in case_results],
        "method_2_proj_cube_3d_ball": [c.method_2_proj_cube_3d_ball for c in case_results],
        "method_3_proj_2d_circles_reconstruct": [c.method_3_proj_2d_circles_reconstruct for c in case_results],
    }
    summary: dict[str, dict] = {}
    for name, rows in methods.items():
        summary[name] = {
            "success_count": int(sum(1 for row in rows if row.success)),
            "time_seconds": summarize([row.elapsed_seconds for row in rows]),
            "abs_diameter_error_mm": summarize([row.abs_diameter_error_mm for row in rows]),
            "abs_stenosis_error_pp": summarize([row.abs_stenosis_error_pp for row in rows]),
        }
    return summary


def case_to_csv_row(case: CaseResult) -> dict:
    row = {
        "case_id": case.case_id,
        "shape_x": case.shape_xyz[0],
        "shape_y": case.shape_xyz[1],
        "shape_z": case.shape_xyz[2],
        "target_diameter_mm": case.target_diameter_mm,
        "target_stenosis_percent": case.target_stenosis_percent,
    }
    for key, result in (
        ("m1", case.method_1_simple_3d_ball),
        ("m2", case.method_2_proj_cube_3d_ball),
        ("m3", case.method_3_proj_2d_circles_reconstruct),
    ):
        row[f"{key}_success"] = int(result.success)
        row[f"{key}_time_s"] = result.elapsed_seconds
        row[f"{key}_diameter_mm"] = result.estimated_diameter_mm
        row[f"{key}_abs_diam_err_mm"] = result.abs_diameter_error_mm
        row[f"{key}_abs_sten_err_pp"] = result.abs_stenosis_error_pp
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=STENOSIS3D_ROOT / "data" / "test" / "masks")
    parser.add_argument("--detector-dir", type=Path, default=STENOSIS3D_ROOT / "detector")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--ratio", type=float, default=0.35)
    parser.add_argument("--half-length-mm", type=float, default=7.0)
    parser.add_argument("--center-mode", type=str, default="middle", choices=["middle", "min_radius"])
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output-json", type=Path, default=ROOT / "temp_stenosis_method_comparison_10cases.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "temp_stenosis_method_comparison_10cases.csv")
    args = parser.parse_args()

    mask_paths = sorted(args.data_dir.glob("*.nii.gz"))[: args.num_cases]
    if len(mask_paths) < args.num_cases:
        raise FileNotFoundError(f"Requested {args.num_cases} masks, found {len(mask_paths)} in {args.data_dir}")

    weights_path = args.detector_dir / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    yolo = YoloBundle(
        model=YOLO(str(weights_path)),
        device=args.device,
        conf=float(args.conf),
        iou=float(args.iou),
    )

    warm_mask_u8, _ = load_mask_nii(mask_paths[0])
    _ = detect_projections_timed(warm_mask_u8, yolo)

    case_results: list[CaseResult] = []
    for mask_path in mask_paths:
        original_mask_u8, pixdim = load_mask_nii(mask_path)
        stenosed_mask_u8, info = simulate_stenosis_pinch(
            original_mask_u8,
            pixdim,
            ratio=args.ratio,
            half_length_mm=args.half_length_mm,
            center_mode=args.center_mode,
        )
        if not info.get("ok", False):
            raise RuntimeError(f"Failed to simulate stenosis for {mask_path.name}: {info}")

        r0_mm = float(info["r0_mm"])
        target_diameter_mm = float(info["target_radius_mm"]) * 2.0
        target_stenosis_percent = max(0.0, (1.0 - (target_diameter_mm / (2.0 * r0_mm))) * 100.0) if r0_mm > 0 else 0.0
        best_boxes, raw_boxes, yolo_elapsed_seconds = detect_projections_timed(stenosed_mask_u8, yolo)

        case_results.append(
            CaseResult(
                case_id=mask_path.name,
                shape_xyz=[int(v) for v in original_mask_u8.shape],
                target_diameter_mm=target_diameter_mm,
                target_stenosis_percent=target_stenosis_percent,
                r0_mm=r0_mm,
                ratio=float(info["ratio"]),
                half_length_mm=float(info["half_length_mm"]),
                method_1_simple_3d_ball=method_simple_3d_ball(
                    stenosed_mask_u8,
                    pixdim,
                    target_diameter_mm,
                    target_stenosis_percent,
                    r0_mm,
                ),
                method_2_proj_cube_3d_ball=method_proj_cube_3d_ball(
                    stenosed_mask_u8,
                    pixdim,
                    target_diameter_mm,
                    target_stenosis_percent,
                    r0_mm,
                    best_boxes,
                    raw_boxes,
                    yolo_elapsed_seconds,
                ),
                method_3_proj_2d_circles_reconstruct=method_proj_2d_circles_reconstruct(
                    stenosed_mask_u8,
                    pixdim,
                    target_diameter_mm,
                    target_stenosis_percent,
                    r0_mm,
                    best_boxes,
                    raw_boxes,
                    yolo_elapsed_seconds,
                ),
            )
        )

    summary = build_summary(case_results)
    output_payload = {
        "config": {
            "data_dir": str(args.data_dir),
            "detector_dir": str(args.detector_dir),
            "num_cases": args.num_cases,
            "ratio": args.ratio,
            "half_length_mm": args.half_length_mm,
            "center_mode": args.center_mode,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
        },
        "summary": summary,
        "cases": [asdict(case) for case in case_results],
    }

    args.output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_rows = [case_to_csv_row(case) for case in case_results]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(json.dumps(output_payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV : {args.output_csv}")


if __name__ == "__main__":
    main()
