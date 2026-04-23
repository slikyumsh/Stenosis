# Stenosis Service

This repository contains the service layer for local coronary stenosis analysis.

## Included components

- `app/` - FastAPI backend, 2D node service, Kafka-based 3D worker, storage and metrics
- `frontend/` - lightweight web UI
- `monitoring/` - Prometheus and Grafana provisioning
- `infra/` - nginx config for 2D upstream routing
- `docker-compose.yml` - local stack
- `Dockerfile` - service image build
- `requirements.app.txt` - runtime dependencies
- `best.onnx` - 2D model artifact tracked with Git LFS

## Features

- synchronous 2D analysis through dedicated 2D nodes
- round-robin balancing across multiple 2D nodes
- optional artifact rendering for faster 2D responses
- asynchronous 3D processing through Kafka and worker services
- local file storage for uploads and inference artifacts
- Prometheus metrics and Grafana dashboards
- custom and academic drift metrics for 2D inputs

## Local run

Use Docker Compose:

```bash
docker compose up --build -d
```

Main endpoints:

- `http://localhost:8080` - frontend
- `http://localhost:8000/docs` - backend API docs
- `http://localhost:9090` - Prometheus
- `http://localhost:3000` - Grafana

Default Grafana credentials:

- user: `admin`
- password: `admin`

## Notes

- The repository intentionally excludes thesis files, example images, old training scripts and generated benchmark outputs.
- Runtime artifacts are stored under `runtime_storage/` and are ignored by git.
