# Local Run

## What is included

- `FastAPI` backend
- `Kafka` queue for async 3D jobs
- local file storage in `runtime_storage/`
- `nginx` frontend on `localhost:8080`
- 2D detection on `best.onnx`
- 2D vessel segmentation as a separate service
- custom 2D drift analysis against the local train reference profile
- 3D async queue with `SegResNet` weights from `../Stenosis3D`

## Run in WSL

```bash
cd /mnt/c/Users/edimv/Desktop/stenosis
docker compose up --build -d
```

## Stop

```bash
cd /mnt/c/Users/edimv/Desktop/stenosis
docker compose down
```

## URLs

- frontend: `http://localhost:8080`
- backend health: `http://localhost:8000/health`
- backend docs: `http://localhost:8000/docs`

## Storage

All uploaded files and generated artifacts are stored locally in:

```text
runtime_storage/
```

## Important parameter

Both analysis endpoints accept `pixel_spacing_mm`.

- default value: `1.0`
- this value is used to convert pixel measurements into millimeters

## Main endpoints

- `POST /api/v1/analyze/2d`
- `POST /api/v1/analyze/3d`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/results/{job_id}`
- `GET /metrics`
