from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import get_settings


def ensure_storage_layout() -> None:
    root = get_settings().storage_root
    for rel_path in [
        "uploads/2d",
        "uploads/3d",
        "results/2d",
        "results/3d",
        "drift",
    ]:
        (root / rel_path).mkdir(parents=True, exist_ok=True)


def new_job_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def save_upload(upload: UploadFile, subdir: str, job_id: str) -> Path:
    filename = upload.filename or ""
    if filename.lower().endswith(".nii.gz"):
        suffix = ".nii.gz"
    else:
        suffix = Path(filename).suffix or ".bin"
    target = get_settings().storage_root / "uploads" / subdir / f"{job_id}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(upload.file.read())
    return target


def make_result_dir(subdir: str, job_id: str) -> Path:
    path = get_settings().storage_root / "results" / subdir / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def to_public_url(path: Path) -> str:
    root = get_settings().storage_root
    rel_path = path.resolve().relative_to(root.resolve())
    normalized = str(rel_path).replace("\\", "/")
    return f"/files/{normalized}"
