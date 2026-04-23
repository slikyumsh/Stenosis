from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.metrics import metrics_middleware, metrics_response
from app.services.detection_2d import get_yolo_model
from app.services.drift_2d import get_reference_profile
from app.services.pipeline_2d import run_2d_pipeline
from app.storage import ensure_storage_layout


class InternalAnalyze2DRequest(BaseModel):
    job_id: str
    input_path: str
    original_filename: str
    pixel_spacing_mm: float = 1.0
    confidence_threshold: float = 0.25
    render_artifacts: bool = True


app = FastAPI(title="Stenosis 2D Node", version="0.1.0")
app.middleware("http")(metrics_middleware)


@app.on_event("startup")
def on_startup() -> None:
    ensure_storage_layout()
    get_yolo_model()
    get_reference_profile()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return metrics_response()


@app.post("/internal/analyze/2d")
def analyze_2d_internal(request: InternalAnalyze2DRequest) -> dict:
    input_path = Path(request.input_path)
    if request.pixel_spacing_mm <= 0:
        raise HTTPException(status_code=400, detail="pixel_spacing_mm must be positive.")
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Input file not found: {input_path}")
    return run_2d_pipeline(
        job_id=request.job_id,
        input_path=input_path,
        original_filename=request.original_filename,
        pixel_spacing_mm=request.pixel_spacing_mm,
        confidence_threshold=request.confidence_threshold,
        render_artifacts=request.render_artifacts,
    )
