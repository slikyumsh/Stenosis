from __future__ import annotations

from itertools import cycle
import json
from pathlib import Path
from threading import Lock

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.db import create_job, get_job, init_db, update_job
from app.metrics import (
    JOB_EVENTS,
    TWO_D_DRIFT_EVENTS,
    TWO_D_DRIFT_SCORE,
    TWO_D_RUNTIME,
    metrics_middleware,
    metrics_response,
)
from app.queue import enqueue_3d_job
from app.storage import ensure_storage_layout, make_result_dir, new_job_id, save_upload, to_public_url
from app.config import get_settings


app = FastAPI(title="Stenosis Local App", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.middleware("http")(metrics_middleware)

_TWO_D_UPSTREAM_URLS = get_settings().twod_node_urls or (get_settings().twod_upstream_url,)
_TWO_D_UPSTREAM_CYCLE = cycle(_TWO_D_UPSTREAM_URLS)
_TWO_D_UPSTREAM_LOCK = Lock()


def _next_twod_upstream_url() -> str:
    with _TWO_D_UPSTREAM_LOCK:
        return next(_TWO_D_UPSTREAM_CYCLE)


@app.on_event("startup")
def on_startup() -> None:
    ensure_storage_layout()
    init_db()


app.mount("/files", StaticFiles(directory=str(get_settings().storage_root), check_dir=False), name="files")


@app.get("/health")
@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return metrics_response()


@app.post("/api/v1/analyze/2d")
def analyze_2d(
    file: UploadFile = File(...),
    pixel_spacing_mm: float = Form(1.0),
    confidence_threshold: float = Form(0.25),
    render_artifacts: bool = Form(True),
):
    if pixel_spacing_mm <= 0:
        raise HTTPException(status_code=400, detail="pixel_spacing_mm must be positive.")

    job_id = new_job_id("2d")
    input_path = save_upload(file, "2d", job_id)
    create_job(
        job_id=job_id,
        job_type="analyze_2d",
        status="running",
        input_path=str(input_path),
        pixel_spacing_mm=pixel_spacing_mm,
        meta={"filename": file.filename or input_path.name},
    )
    JOB_EVENTS.labels("2d", "running").inc()
    request_payload = {
        "job_id": job_id,
        "input_path": str(input_path),
        "original_filename": file.filename or input_path.name,
        "pixel_spacing_mm": pixel_spacing_mm,
        "confidence_threshold": confidence_threshold,
        "render_artifacts": render_artifacts,
    }

    try:
        upstream_url = _next_twod_upstream_url()
        with httpx.Client(timeout=300.0) as client:
            upstream_response = client.post(upstream_url, json=request_payload)
        upstream_response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text or "2D upstream request failed."
        update_job(job_id, status="failed", error_text=detail)
        JOB_EVENTS.labels("2d", "failed").inc()
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.HTTPError as exc:
        detail = f"2D upstream is unavailable: {exc}"
        update_job(job_id, status="failed", error_text=detail)
        JOB_EVENTS.labels("2d", "failed").inc()
        raise HTTPException(status_code=502, detail=detail) from exc

    result_payload = upstream_response.json()
    result_path = make_result_dir("2d", job_id) / "result.json"
    update_job(job_id, status="completed", result_path=str(result_path), meta={"filename": file.filename or input_path.name})
    JOB_EVENTS.labels("2d", "completed").inc()
    TWO_D_RUNTIME.observe(float(result_payload["detection"]["inference_seconds"]))
    TWO_D_DRIFT_SCORE.observe(float(result_payload["drift"]["combined_score"]))
    TWO_D_DRIFT_EVENTS.labels(result_payload["drift"]["level"]).inc()
    return result_payload


@app.post("/api/v1/analyze/3d")
def analyze_3d(
    file: UploadFile = File(...),
    pixel_spacing_mm: float = Form(1.0),
):
    if pixel_spacing_mm <= 0:
        raise HTTPException(status_code=400, detail="pixel_spacing_mm must be positive.")

    job_id = new_job_id("3d")
    input_path = save_upload(file, "3d", job_id)
    create_job(
        job_id=job_id,
        job_type="analyze_3d",
        status="queued",
        input_path=str(input_path),
        pixel_spacing_mm=pixel_spacing_mm,
        meta={"filename": file.filename or input_path.name},
    )
    enqueue_3d_job(job_id)
    JOB_EVENTS.labels("3d", "queued").inc()
    return {
        "job_id": job_id,
        "status": "queued",
        "pixel_spacing_mm": pixel_spacing_mm,
        "input_url": to_public_url(input_path),
    }


@app.get("/api/v1/jobs/{job_id}")
def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    response = dict(job)
    input_path = Path(job["input_path"])
    response["input_url"] = to_public_url(input_path) if input_path.exists() else None
    if job.get("result_path"):
        result_path = Path(job["result_path"])
        response["result_url"] = to_public_url(result_path) if result_path.exists() else None
    return response


@app.get("/api/v1/results/{job_id}")
def get_result(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not job.get("result_path"):
        raise HTTPException(status_code=409, detail=f"Job is not completed. Current status: {job['status']}")
    result_path = Path(job["result_path"])
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found.")
    with result_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
