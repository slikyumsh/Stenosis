from __future__ import annotations

import time

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


HTTP_REQUESTS = Counter(
    "stenosis_http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "stenosis_http_request_latency_seconds",
    "HTTP request latency.",
    ["method", "path"],
)
TWO_D_RUNTIME = Histogram(
    "stenosis_2d_inference_seconds",
    "2D analysis runtime.",
)
TWO_D_DRIFT_SCORE = Histogram(
    "stenosis_2d_drift_score",
    "Custom 2D drift score.",
)
TWO_D_DRIFT_EVENTS = Counter(
    "stenosis_2d_drift_events_total",
    "2D drift level events.",
    ["level"],
)
THREE_D_RUNTIME = Histogram(
    "stenosis_3d_inference_seconds",
    "3D analysis runtime.",
)
JOB_EVENTS = Counter(
    "stenosis_jobs_total",
    "Background job events.",
    ["job_type", "status"],
)


async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    HTTP_LATENCY.labels(request.method, path).observe(elapsed)
    return response


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
