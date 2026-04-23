from __future__ import annotations

import logging
import time
from pathlib import Path

from app.db import get_job, init_db, update_job
from app.metrics import JOB_EVENTS, THREE_D_RUNTIME
from app.queue import acknowledge_3d_job, blocking_pop_3d_job
from app.services.segmentation_3d import process_volume
from app.storage import ensure_storage_layout, make_result_dir, to_public_url, write_json

LOGGER = logging.getLogger("stenosis.worker")


def process_job(job_id: str) -> None:
    job = get_job(job_id)
    if job is None:
        return

    update_job(job_id, status="running")
    JOB_EVENTS.labels("3d", "running").inc()
    started = time.perf_counter()
    try:
        input_path = Path(job["input_path"])
        result_dir = make_result_dir("3d", job_id)
        payload = process_volume(input_path, result_dir, pixel_spacing_mm=float(job["pixel_spacing_mm"]))
        elapsed = time.perf_counter() - started

        public_artifacts = {
            name: to_public_url(path)
            for name, path in payload["artifacts"].items()
        }
        result_payload = {
            "job_id": job_id,
            "job_type": "analyze_3d",
            "status": "completed",
            "pixel_spacing_mm": float(job["pixel_spacing_mm"]),
            "input": {
                "filename": job["meta"].get("filename", input_path.name),
                "volume_url": to_public_url(input_path),
            },
            "geometry": payload["geometry"],
            "artifacts": public_artifacts,
            "inference_seconds": elapsed,
        }

        result_path = result_dir / "result.json"
        write_json(result_path, result_payload)
        update_job(job_id, status="completed", result_path=str(result_path))
        JOB_EVENTS.labels("3d", "completed").inc()
        THREE_D_RUNTIME.observe(elapsed)
    except Exception as exc:  # pragma: no cover - defensive worker guard
        update_job(job_id, status="failed", error_text=str(exc))
        JOB_EVENTS.labels("3d", "failed").inc()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ensure_storage_layout()
    init_db()
    while True:
        try:
            job_id = blocking_pop_3d_job(timeout=5)
            if job_id is None:
                continue
            process_job(job_id)
            acknowledge_3d_job()
        except Exception as exc:  # pragma: no cover - resilience loop
            LOGGER.exception("Worker loop error: %s", exc)
            time.sleep(2)


if __name__ == "__main__":
    main()
