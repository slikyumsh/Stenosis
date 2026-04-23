from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    storage_root: Path
    db_path: Path
    node_name: str
    twod_upstream_url: str
    twod_node_urls: tuple[str, ...]
    kafka_bootstrap_servers: str
    kafka_topic: str
    kafka_consumer_group: str
    drift_reference_glob: str
    drift_reference_sample_limit: int
    yolo_model_path: Path
    segresnet_checkpoint_path: Path
    torch_device: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    storage_root = Path(os.getenv("APP_STORAGE_ROOT", "runtime_storage")).resolve()
    db_path = Path(os.getenv("APP_DB_PATH", storage_root / "app.db")).resolve()
    twod_node_urls_raw = os.getenv("APP_2D_NODE_URLS", "")
    twod_node_urls = tuple(url.strip() for url in twod_node_urls_raw.split(",") if url.strip())
    return Settings(
        storage_root=storage_root,
        db_path=db_path,
        node_name=os.getenv("APP_NODE_NAME", "api-gateway"),
        twod_upstream_url=os.getenv("APP_2D_UPSTREAM_URL", "http://localhost:8000/internal/analyze/2d"),
        twod_node_urls=twod_node_urls,
        kafka_bootstrap_servers=os.getenv("APP_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        kafka_topic=os.getenv("APP_KAFKA_TOPIC", "stenosis-jobs-3d"),
        kafka_consumer_group=os.getenv("APP_KAFKA_CONSUMER_GROUP", "stenosis-worker"),
        drift_reference_glob=os.getenv("APP_DRIFT_REFERENCE_GLOB", "data/train/*.bmp"),
        drift_reference_sample_limit=int(os.getenv("APP_DRIFT_REFERENCE_SAMPLE_LIMIT", "12")),
        yolo_model_path=Path(os.getenv("APP_YOLO_MODEL_PATH", "best.onnx")).resolve(),
        segresnet_checkpoint_path=Path(
            os.getenv(
                "APP_SEGRESNET_CHECKPOINT_PATH",
                "../Stenosis3D/models/SegResNet/runs/20260114_183930/checkpoints/best.pt",
            )
        ).resolve(),
        torch_device=os.getenv("APP_TORCH_DEVICE", "cpu"),
    )
