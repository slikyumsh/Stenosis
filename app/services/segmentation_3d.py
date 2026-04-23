from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from app.config import get_settings


def _load_run_config() -> dict:
    checkpoint_path = get_settings().segresnet_checkpoint_path
    config_path = checkpoint_path.parent.parent / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "roi": [96, 96, 96],
        "sw_batch_size": 1,
        "sw_overlap": 0.25,
        "amp": False,
    }


@lru_cache(maxsize=1)
def get_3d_model_bundle() -> tuple[torch.nn.Module, dict, torch.device]:
    settings = get_settings()
    cfg = _load_run_config()
    device = torch.device(settings.torch_device)
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.1,
    ).to(device)

    payload = torch.load(str(settings.segresnet_checkpoint_path), map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, cfg, device


def _load_volume(input_path: Path) -> tuple[np.ndarray, np.ndarray]:
    nii = nib.load(str(input_path))
    volume = nii.get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("Only 3D volumes are supported.")
    affine = nii.affine
    return volume, affine


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    mask = volume != 0
    if not np.any(mask):
        return volume
    values = volume[mask]
    mean = float(values.mean())
    std = float(values.std()) or 1.0
    out = volume.copy()
    out[mask] = (out[mask] - mean) / std
    return out


def _pad_to_roi(tensor: torch.Tensor, roi: list[int]) -> torch.Tensor:
    _, _, depth, height, width = tensor.shape
    target_depth = max(depth, roi[0])
    target_height = max(height, roi[1])
    target_width = max(width, roi[2])
    pad_d = target_depth - depth
    pad_h = target_height - height
    pad_w = target_width - width
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d))


def _save_preview(mask: np.ndarray, output_path: Path) -> None:
    mip = mask.max(axis=2).astype(np.uint8) * 255
    cv2.imwrite(str(output_path), mip)


def _save_centerline_csv(centerline: np.ndarray, diameter_mm: np.ndarray, output_path: Path) -> None:
    coords = np.argwhere(centerline > 0)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z", "diameter_mm"])
        for x, y, z in coords:
            writer.writerow([int(x), int(y), int(z), float(diameter_mm[x, y, z])])


def _geometry_stats(mask: np.ndarray, pixel_spacing_mm: float) -> tuple[np.ndarray, np.ndarray, dict]:
    if mask.max() == 0:
        empty = np.zeros_like(mask, dtype=np.float32)
        return empty.astype(np.uint8), empty, {
            "mask_voxels": 0,
            "mask_volume_mm3": 0.0,
            "centerline_voxels": 0,
            "centerline_length_mm": 0.0,
            "min_diameter_mm": 0.0,
            "mean_diameter_mm": 0.0,
            "max_diameter_mm": 0.0,
            "estimated_stenosis_size_mm": 0.0,
            "stenosis_ratio_percent": 0.0,
            "stenosis_narrowing_percent": 0.0,
        }

    centerline = skeletonize(mask.astype(bool), method="lee").astype(np.uint8)
    edt = distance_transform_edt(mask.astype(bool), sampling=(pixel_spacing_mm,) * 3).astype(np.float32)
    diameters = edt * 2.0
    centerline_diameters = diameters[centerline > 0]
    if centerline_diameters.size == 0:
        centerline_diameters = np.array([0.0], dtype=np.float32)

    min_diameter = float(centerline_diameters.min())
    mean_diameter = float(centerline_diameters.mean())
    max_diameter = float(centerline_diameters.max())
    ratio = (min_diameter / max_diameter) * 100.0 if max_diameter > 0 else 0.0
    ratio = max(0.0, min(100.0, ratio))

    stats = {
        "mask_voxels": int(mask.sum()),
        "mask_volume_mm3": round(float(mask.sum() * (pixel_spacing_mm ** 3)), 2),
        "centerline_voxels": int(centerline.sum()),
        "centerline_length_mm": round(float(centerline.sum() * pixel_spacing_mm), 2),
        "min_diameter_mm": round(min_diameter, 2),
        "mean_diameter_mm": round(mean_diameter, 2),
        "max_diameter_mm": round(max_diameter, 2),
        "estimated_stenosis_size_mm": round(min_diameter, 2),
        "stenosis_ratio_percent": round(ratio, 2),
        "stenosis_narrowing_percent": round(100.0 - ratio, 2),
    }
    return centerline, diameters, stats


def process_volume(input_path: Path, output_dir: Path, pixel_spacing_mm: float = 1.0) -> dict:
    model, cfg, device = get_3d_model_bundle()
    volume, affine = _load_volume(input_path)
    normalized = _normalize_volume(volume)
    tensor = torch.from_numpy(normalized[None, None, ...]).float()
    tensor = _pad_to_roi(tensor, cfg.get("roi", [96, 96, 96])).to(device)

    use_amp = bool(cfg.get("amp", False) and device.type == "cuda")
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type if device.type == "cuda" else "cpu", enabled=use_amp):
            logits = sliding_window_inference(
                inputs=tensor,
                roi_size=cfg.get("roi", [96, 96, 96]),
                sw_batch_size=int(cfg.get("sw_batch_size", 1)),
                predictor=model,
                overlap=float(cfg.get("sw_overlap", 0.25)),
                mode="gaussian",
                sw_device=device,
                device=device,
                progress=False,
            )

    prediction = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    prediction = prediction[: volume.shape[0], : volume.shape[1], : volume.shape[2]]

    centerline, diameters, geometry = _geometry_stats(prediction, pixel_spacing_mm)

    mask_path = output_dir / "pred_mask.nii.gz"
    centerline_path = output_dir / "centerline.nii.gz"
    centerline_csv_path = output_dir / "centerline.csv"
    preview_path = output_dir / "preview_mip.png"

    nib.save(nib.Nifti1Image(prediction.astype(np.uint8), affine), str(mask_path))
    nib.save(nib.Nifti1Image(centerline.astype(np.uint8), affine), str(centerline_path))
    _save_centerline_csv(centerline, diameters, centerline_csv_path)
    _save_preview(prediction, preview_path)

    return {
        "geometry": geometry,
        "artifacts": {
            "pred_mask": mask_path,
            "centerline_mask": centerline_path,
            "centerline_csv": centerline_csv_path,
            "preview_mip": preview_path,
        },
    }
