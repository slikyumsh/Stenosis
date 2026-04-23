from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.services.analysis_2d import bbox_measurements, local_vessel_stats
from app.services.detection_2d import run_detection, save_detection_overlay
from app.services.drift_2d import analyze_request_drift
from app.services.segmentation_2d import (
    global_vessel_stats,
    save_mask,
    save_mask_render,
    save_segmentation_overlay,
    segment_vessels,
)
from app.storage import make_result_dir, to_public_url, write_json


def run_2d_pipeline(
    *,
    job_id: str,
    input_path: Path,
    original_filename: str,
    pixel_spacing_mm: float,
    confidence_threshold: float,
    render_artifacts: bool,
) -> dict:
    result_dir = make_result_dir("2d", job_id)
    detection = run_detection(input_path, confidence_threshold=confidence_threshold)
    segmentation = segment_vessels(input_path)
    drift_payload = analyze_request_drift(input_path, segmentation["mask"], detection["boxes"])

    detection_overlay = result_dir / "detection_overlay.png"
    mask_path = result_dir / "vessel_mask.png"
    mask_render_path = result_dir / "vessel_mask_render.png"
    segmentation_overlay = result_dir / "segmentation_overlay.png"

    if render_artifacts:
        save_detection_overlay(detection["image"], detection["boxes"], detection_overlay)
        save_mask(segmentation["mask"], mask_path)
        save_mask_render(segmentation["mask"], mask_render_path)
        save_segmentation_overlay(segmentation["gray"], segmentation["mask"], segmentation_overlay)

    boxes_with_metrics = []
    for box in bbox_measurements(detection["boxes"], pixel_spacing_mm):
        boxes_with_metrics.append({**box, **local_vessel_stats(segmentation["mask"], box, pixel_spacing_mm)})

    vessel_stats = global_vessel_stats(segmentation["mask"], pixel_spacing_mm)
    result_payload = {
        "job_id": job_id,
        "job_type": "analyze_2d",
        "status": "completed",
        "pixel_spacing_mm": pixel_spacing_mm,
        "render_artifacts": render_artifacts,
        "processing_node": get_settings().node_name,
        "input": {
            "filename": original_filename or input_path.name,
            "image_url": to_public_url(input_path),
        },
        "detection": {
            "image_width": detection["width"],
            "image_height": detection["height"],
            "boxes": boxes_with_metrics,
            "inference_seconds": detection["inference_seconds"],
        },
        "segmentation": {
            **vessel_stats,
            "mask_url": to_public_url(mask_path) if render_artifacts else None,
            "mask_render_url": to_public_url(mask_render_path) if render_artifacts else None,
            "overlay_url": to_public_url(segmentation_overlay) if render_artifacts else None,
        },
        "drift": drift_payload,
        "artifacts": {
            "detection_overlay_url": to_public_url(detection_overlay) if render_artifacts else None,
        },
    }
    result_path = result_dir / "result.json"
    write_json(result_path, result_payload)
    return result_payload
