"""FastAPI service: YOLOv8n object detection on ONNX Runtime (CPU)."""
from __future__ import annotations

import io
import json
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import metrics
from app.config import settings
from app.detector import Detection, Detector
from app.errors import APIError
from app.jobs import get_queue
from app.schemas import BatchItem, BatchResponse, DetectResponse, HealthResponse
from app.tracker import IoUTracker

STATIC = Path(__file__).parent / "static"
detector: Detector | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global detector
    detector = Detector(settings.model_path, imgsz=settings.imgsz, threads=settings.ort_threads)
    detector.detect([np.zeros((settings.imgsz, settings.imgsz, 3), np.uint8)])  # warm-up
    metrics.MODEL_LOADED.set(1)
    get_queue(_get_detector)
    yield
    metrics.MODEL_LOADED.set(0)


app = FastAPI(
    title="YOLOv8 Detection API",
    version="1.0.0",
    description="Object detection with pretrained YOLOv8n (COCO, 80 classes) served on ONNX Runtime.",
    lifespan=lifespan,
)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(APIError)
async def api_error(_: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(exc.body(), status_code=exc.status_code)


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
    # Serverless runtimes (Vercel) may skip the ASGI lifespan, so load lazily as a fallback.
    global detector
    if detector is None:
        detector = Detector(settings.model_path, imgsz=settings.imgsz, threads=settings.ort_threads)
        metrics.MODEL_LOADED.set(1)
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


VIDEO_TYPES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def _tracked(tracked) -> list[dict]:
    return [{"track_id": t.track_id, **item} for t, item in zip(tracked, _serialize([t.detection for t in tracked]))]


@app.post("/detect/video", summary="Detect and track objects in a short video clip")
def detect_video(
    file: UploadFile = File(..., description="MP4/WebM/MOV clip"),
    sample_fps: float = Query(5.0, gt=0, le=30, description="Frames per second of video to analyse"),
    max_frames: int = Query(60, ge=1, le=settings.max_video_frames, description="Upper bound on analysed frames"),
    conf: float | None = ConfQ,
    iou: float | None = IouQ,
    track: bool = Query(True, description="Link detections across frames with the IoU tracker"),
):
    import cv2

    model = _get_detector()
    data = file.file.read(int(settings.max_upload_mb * 1024 * 1024) + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"{file.filename}: larger than {settings.max_upload_mb:g} MB")
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".mp4"
    if suffix not in VIDEO_TYPES:
        raise HTTPException(415, f"{file.filename}: unsupported video type {suffix}")
    start = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise HTTPException(415, f"{file.filename}: could not decode video")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, round(src_fps / sample_fps))
        frames, indices, idx = [], [], 0
        while len(frames) < max_frames:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, bgr = cap.retrieve()
                if not ok:
                    break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                indices.append(idx)
            idx += 1
        cap.release()
    if not frames:
        raise HTTPException(415, f"{file.filename}: no decodable frames")
    decode_ms = (time.perf_counter() - start) * 1000

    tracker = IoUTracker() if track else None
    out_frames, inference_ms = [], []
    chunk = max(1, settings.max_batch)
    for c in range(0, len(frames), chunk):
        batch = frames[c : c + chunk]
        results, timings = model.detect(
            batch,
            conf_threshold=conf if conf is not None else settings.conf_threshold,
            iou_threshold=iou if iou is not None else settings.iou_threshold,
            max_detections=settings.max_detections,
        )
        inference_ms.append(timings.inference_ms / len(batch))
        metrics.INFERENCE_LATENCY.labels("/detect/video").observe(timings.inference_ms / 1000)
        for j, dets in enumerate(results):
            fi = indices[c + j]
            if tracker:
                items = _tracked(tracker.update(dets))
            else:
                items = _serialize(dets)
            out_frames.append({"frame": fi, "t": round(fi / src_fps, 3), "count": len(items), "detections": items})
    metrics.IMAGES.labels("/detect/video").inc(len(frames))
    elapsed = (time.perf_counter() - start) * 1000
    h, w = frames[0].shape[:2]
    return {
        "model": model.model_name,
        "filename": file.filename,
        "video": {"width": w, "height": h, "fps": round(src_fps, 2), "frames": total, "duration_s": round(total / src_fps, 2)},
        "frames_analysed": len(frames),
        "frame_step": step,
        "tracking": bool(tracker),
        "unique_objects": tracker.summary() if tracker else None,
        "frames": out_frames,
        "timings": {
            "decode_ms": round(decode_ms, 1),
            "total_ms": round(elapsed, 1),
            "inference_ms_per_frame": round(float(np.mean(inference_ms)), 2),
            "processing_fps": round(len(frames) / (elapsed / 1000), 2),
        },
    }


@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    """Streaming mode for webcams: send JPEG/PNG frames as binary messages, get JSON back per frame.

    Tracking state lives for the life of the connection. Send a text message
    like {"conf": 0.4} to change the threshold mid-stream. Needs a long-lived
    server (Docker / uvicorn); serverless hosts like Vercel do not keep WebSockets open.
    """
    await ws.accept()
    model = _get_detector()
    tracker = IoUTracker()
    conf = settings.conf_threshold
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text"):
                try:
                    conf = float(json.loads(msg["text"]).get("conf", conf))
                except (ValueError, AttributeError):
                    await ws.send_json({"error": "expected JSON like {\"conf\": 0.4}"})
                continue
            raw = msg.get("bytes")
            if not raw:
                continue
            try:
                with Image.open(io.BytesIO(raw)) as img:
                    image = np.asarray(img.convert("RGB"))
            except (UnidentifiedImageError, OSError):
                await ws.send_json({"error": "frame is not a decodable image"})
                continue
            (dets,), timings = model.detect([image], conf_threshold=conf, iou_threshold=settings.iou_threshold,
                                            max_detections=settings.max_detections)
            metrics.INFERENCE_LATENCY.labels("/ws/detect").observe(timings.inference_ms / 1000)
            metrics.IMAGES.labels("/ws/detect").inc()
            await ws.send_json({
                "frame": tracker._frame,
                "detections": _tracked(tracker.update(dets)),
                "unique_objects": tracker.summary(),
                "timings": timings.__dict__,
            })
    except WebSocketDisconnect:
        pass


def _unit(name: str, value: float | None, default: float) -> float:
    resolved = default if value is None else value
    if not 0.0 <= resolved <= 1.0:
        raise APIError(422, f"invalid_{name}", f"{name} must be between 0 and 1.", {name: resolved})
    return resolved


@app.post("/jobs", status_code=202, summary="Queue a short clip for annotated detection")
async def create_job(
    file: UploadFile = File(..., description="MP4, WebM, MOV, or AVI up to 10 seconds"),
    conf: float | None = Form(None),
    iou: float | None = Form(None),
):
    """Background job. One clip runs at a time. Poll `GET /jobs/{job_id}` then fetch the MP4."""
    data = await file.read(int(settings.max_upload_mb * 1024 * 1024) + 1)
    queue = get_queue(_get_detector)
    job = queue.submit(
        data,
        file.filename or "clip.mp4",
        file.content_type,
        _unit("conf", conf, settings.conf_threshold),
        _unit("iou", iou, settings.iou_threshold),
    )
    return queue.get(job.id)


@app.get("/jobs/{job_id}", summary="Job status, progress, and per-frame detections")
def job_status(job_id: str):
    return get_queue(_get_detector).get(job_id)


@app.get("/jobs/{job_id}/video", summary="Annotated MP4 for a finished job")
def job_video(job_id: str) -> FileResponse:
    path = get_queue(_get_detector).video_path(job_id)
    return FileResponse(path, media_type="video/mp4", filename="annotated.mp4", content_disposition_type="inline")


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


@app.get("/ready", summary="Readiness: the ONNX session is loaded")
def ready():
    if detector is None:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return {"status": "ready", "model": detector.model_name, "imgsz": detector.imgsz}


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
