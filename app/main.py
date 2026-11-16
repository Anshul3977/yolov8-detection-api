"""FastAPI service: YOLOv8n object detection on ONNX Runtime (CPU)."""
from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import metrics
from app.config import settings
from app.detector import Detection, Detector
from app.schemas import BatchItem, BatchResponse, DetectResponse, HealthResponse

STATIC = Path(__file__).parent / "static"
detector: Detector | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global detector
    detector = Detector(settings.model_path, imgsz=settings.imgsz, threads=settings.ort_threads)
    detector.detect([np.zeros((settings.imgsz, settings.imgsz, 3), np.uint8)])  # warm-up
    metrics.MODEL_LOADED.set(1)
    yield
    metrics.MODEL_LOADED.set(0)


app = FastAPI(
    title="YOLOv8 Detection API",
    version="1.0.0",
    description="Object detection with pretrained YOLOv8n (COCO, 80 classes) served on ONNX Runtime.",
    lifespan=lifespan,
)


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = route.path if route else "unmatched"
    if path != "/metrics":
        metrics.REQUESTS.labels(request.method, path, response.status_code).inc()
        metrics.REQUEST_LATENCY.labels(path).observe(time.perf_counter() - start)
    return response


def _get_detector() -> Detector:
    if detector is None:
        raise HTTPException(503, "Model not loaded yet")
    return detector


def _decode(upload: UploadFile) -> np.ndarray:
    data = upload.file.read(int(settings.max_upload_mb * 1024 * 1024) + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"{upload.filename}: larger than {settings.max_upload_mb:g} MB")
    if not data:
        raise HTTPException(400, f"{upload.filename}: empty file")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            return np.asarray(img)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(415, f"{upload.filename}: not a decodable image") from exc


def _serialize(dets: list[Detection]) -> list[dict]:
    for d in dets:
        metrics.DETECTIONS.labels(d.label).inc()
    return [
        {
            "class_id": d.class_id,
            "label": d.label,
            "confidence": d.confidence,
            "box": {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2},
        }
        for d in dets
    ]


ConfQ = Query(None, ge=0.0, le=1.0, description="Confidence threshold (default from CONF_THRESHOLD)")
IouQ = Query(None, ge=0.0, le=1.0, description="NMS IoU threshold (default from IOU_THRESHOLD)")
MaxDetQ = Query(None, ge=1, le=1000, description="Max detections per image")


@app.post("/detect", response_model=DetectResponse, summary="Detect objects in one image")
def detect(
    file: UploadFile = File(..., description="JPEG/PNG/WebP image"),
    conf: float | None = ConfQ,
    iou: float | None = IouQ,
    max_det: int | None = MaxDetQ,
):
    model = _get_detector()
    image = _decode(file)
    (dets,), timings = model.detect(
        [image],
        conf_threshold=conf if conf is not None else settings.conf_threshold,
        iou_threshold=iou if iou is not None else settings.iou_threshold,
        max_detections=max_det or settings.max_detections,
    )
    metrics.INFERENCE_LATENCY.labels("/detect").observe(timings.inference_ms / 1000)
    metrics.IMAGES.labels("/detect").inc()
    h, w = image.shape[:2]
    return {
        "model": model.model_name,
        "filename": file.filename,
        "image": {"width": w, "height": h},
        "count": len(dets),
        "detections": _serialize(dets),
        "timings": timings.__dict__,
    }


@app.post("/detect/batch", response_model=BatchResponse, summary="Detect objects in several images (one forward pass)")
def detect_batch(
    files: list[UploadFile] = File(..., description=f"Up to MAX_BATCH images (default {settings.max_batch})"),
    conf: float | None = ConfQ,
    iou: float | None = IouQ,
    max_det: int | None = MaxDetQ,
):
    model = _get_detector()
    if len(files) > settings.max_batch:
        raise HTTPException(413, f"At most {settings.max_batch} images per batch, got {len(files)}")
    images = [_decode(f) for f in files]
    results, timings = model.detect(
        images,
        conf_threshold=conf if conf is not None else settings.conf_threshold,
        iou_threshold=iou if iou is not None else settings.iou_threshold,
        max_detections=max_det or settings.max_detections,
    )
    metrics.INFERENCE_LATENCY.labels("/detect/batch").observe(timings.inference_ms / 1000)
    metrics.IMAGES.labels("/detect/batch").inc(len(images))
    metrics.BATCH_SIZE.observe(len(images))
    items = [
        BatchItem(
            filename=f.filename,
            image={"width": img.shape[1], "height": img.shape[0]},
            count=len(dets),
            detections=_serialize(dets),
        )
        for f, img, dets in zip(files, images, results)
    ]
    return {"model": model.model_name, "batch_size": len(images), "results": items, "timings": timings.__dict__}


@app.get("/health", response_model=HealthResponse, summary="Liveness and model status")
def health():
    model = _get_detector()
    return {
        "status": "ok",
        "model": model.model_name,
        "providers": model.session.get_providers(),
        "classes": len(model.names),
        "imgsz": model.imgsz,
    }


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
