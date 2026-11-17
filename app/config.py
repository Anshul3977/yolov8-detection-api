"""Runtime settings, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", ROOT / "models" / "yolov8n.onnx"))
    imgsz: int = _int("IMGSZ", 640)
    conf_threshold: float = _float("CONF_THRESHOLD", 0.25)
    iou_threshold: float = _float("IOU_THRESHOLD", 0.45)
    max_detections: int = _int("MAX_DETECTIONS", 300)
    max_batch: int = _int("MAX_BATCH", 16)
    max_upload_mb: float = _float("MAX_UPLOAD_MB", 10)
    max_video_frames: int = _int("MAX_VIDEO_FRAMES", 300)
    max_video_seconds: float = _float("MAX_VIDEO_SECONDS", 10)
    max_queued_jobs: int = _int("MAX_QUEUED_JOBS", 4)
    # Finished clip directories kept on disk. The upload and the raw AVI are
    # deleted as soon as the job ends; only the annotated MP4 stays, and only
    # this many of them.
    max_retained_jobs: int = _int("MAX_RETAINED_JOBS", 2)
    job_ttl_seconds: int = _int("JOB_TTL_SECONDS", 1800)
    # 0 lets ONNX Runtime pick (one thread per physical core).
    ort_threads: int = _int("ORT_THREADS", 0)


settings = Settings()
