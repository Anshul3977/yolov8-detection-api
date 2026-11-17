"""In-process queue for short-clip jobs. One clip runs at a time."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.config import settings
from app.errors import APIError

logger = logging.getLogger("app.jobs")

VIDEO_MAGIC = {
    "mp4": "video/mp4",
    "webm": "video/webm",
    "avi": "video/x-msvideo",
}
DECLARED_OK = {
    "video/mp4": {"video/mp4", "video/quicktime", "application/mp4"},
    "video/webm": {"video/webm"},
    "video/x-msvideo": {"video/x-msvideo", "video/avi", "video/msvideo"},
}


def sniff_video(data: bytes) -> str | None:
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    if data.startswith(b"RIFF") and data[8:12] == b"AVI ":
        return "video/x-msvideo"
    return None


def _check_declared(sniffed: str, content_type: str | None) -> None:
    declared = (content_type or "").split(";", 1)[0].strip().lower()
    if not declared or declared in {"application/octet-stream", "binary/octet-stream"}:
        return
    allowed = DECLARED_OK.get(sniffed, {sniffed})
    if declared not in allowed:
        raise APIError(
            415,
            "media_type_mismatch",
            "The Content-Type does not match the file contents.",
            {"declared": declared, "detected": sniffed},
        )


def _draw(frame: np.ndarray, detections) -> np.ndarray:
    import cv2

    canvas = frame.copy()
    for det in detections:
        color = (40 + (det.class_id * 47) % 180, 180, 255)
        p1 = (int(round(det.x1)), int(round(det.y1)))
        p2 = (int(round(det.x2)), int(round(det.y2)))
        cv2.rectangle(canvas, p1, p2, color, 2)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (p1[0], max(14, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _transcode(src: Path, dest: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise APIError(500, "ffmpeg_missing", "ffmpeg is required to encode a browser-playable MP4.")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not dest.is_file():
        logger.error("ffmpeg failed: %s", proc.stderr[-500:])
        raise APIError(500, "video_encode_failed", "ffmpeg could not encode the annotated clip.")


def _probe_limits(path: Path) -> tuple[float, int, float, int, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise APIError(400, "unreadable_video", "OpenCV could not open this clip.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frames <= 0 or width < 2 or height < 2:
        raise APIError(400, "unreadable_video", "Could not read the clip duration or frame size.")
    duration = frames / fps
    if duration > settings.max_video_seconds + 1e-3:
        raise APIError(
            422,
            "video_too_long",
            f"Clips must be {settings.max_video_seconds:g} seconds or shorter.",
            {"duration_s": round(duration, 3), "limit_s": settings.max_video_seconds},
        )
    if frames > settings.max_video_frames:
        raise APIError(
            422,
            "too_many_frames",
            "The clip has more frames than the server will process.",
            {"frames": frames, "limit": settings.max_video_frames},
        )
    if max(width, height) > 1920:
        raise APIError(
            422,
            "video_resolution_too_high",
            "Scale the clip down to 1920px on the long side.",
            {"width": width, "height": height, "max_side": 1920},
        )
    return fps, frames, duration, width, height


@dataclass
class Job:
    id: str
    filename: str
    conf: float
    iou: float
    created_at: float
    work_dir: Path
    status: str = "queued"
    progress: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    error: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    video_path: Path | None = None
    input_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.input_path = self.work_dir / "input"

    def view(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "job_id": self.id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "filename": self.filename,
            "conf": self.conf,
            "iou": self.iou,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
        }
        if self.status == "succeeded":
            body["video_url"] = f"/jobs/{self.id}/video"
        return body


class JobQueue:
    def __init__(self, detector_factory: Callable[[], Any]) -> None:
        self._detector_factory = detector_factory
        self._jobs: dict[str, Job] = {}
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._root = Path(tempfile.gettempdir()) / "yolov8-jobs"
        self._root.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="clip-worker", daemon=True)
        self._thread.start()

    def submit(self, data: bytes, filename: str, content_type: str | None, conf: float, iou: float) -> Job:
        limit = int(settings.max_upload_mb * 1024 * 1024)
        if len(data) > limit:
            raise APIError(
                413,
                "file_too_large",
                "The clip exceeds the upload limit.",
                {"limit_bytes": limit, "received_bytes": len(data)},
            )
        if not data:
            raise APIError(400, "empty_file", "The upload is empty.")
        sniffed = sniff_video(data)
        if sniffed is None:
            raise APIError(415, "unsupported_media", "Upload an MP4, WebM, MOV, or AVI clip.")
        _check_declared(sniffed, content_type)
        if not 0.0 <= conf <= 1.0:
            raise APIError(422, "invalid_confidence", "conf must be between 0 and 1.", {"conf": conf})
        if not 0.0 <= iou <= 1.0:
            raise APIError(422, "invalid_iou", "iou must be between 0 and 1.", {"iou": iou})

        job_id = uuid.uuid4().hex
        work = self._root / job_id
        work.mkdir(parents=True)
        job = Job(
            id=job_id,
            filename=Path(filename or "clip.mp4").name[:120],
            conf=conf,
            iou=iou,
            created_at=time.time(),
            work_dir=work,
        )
        job.input_path.write_bytes(data)
        try:
            _probe_limits(job.input_path)
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise
        with self._lock:
            queued = sum(1 for item in self._jobs.values() if item.status == "queued")
            if queued >= settings.max_queued_jobs:
                shutil.rmtree(work, ignore_errors=True)
                raise APIError(
                    429,
                    "queue_full",
                    "The clip queue is full. Wait for the running job to finish.",
                    {"max_queued_jobs": settings.max_queued_jobs},
                )
            self._jobs[job_id] = job
            self._queue.append(job_id)
        self._wake.set()
        return job

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise APIError(404, "job_not_found", "No job with that id.")
            return job.view()

    def video_path(self, job_id: str) -> Path:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise APIError(404, "job_not_found", "No job with that id.")
            if job.status != "succeeded" or job.video_path is None:
                raise APIError(409, "job_not_ready", "The annotated clip is not ready.", {"status": job.status})
            path = job.video_path
        if not path.is_file():
            raise APIError(404, "video_missing", "The annotated clip is no longer on disk.")
        return path

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            while True:
                with self._lock:
                    if not self._queue:
                        break
                    job_id = self._queue.pop(0)
                try:
                    self._run(job_id)
                except Exception:
                    logger.exception("clip job %s crashed", job_id)
                self._sweep()

    def _run(self, job_id: str) -> None:
        import cv2

        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = time.time()
        cap = None
        writer = None
        try:
            fps, frame_count, duration, width, height = _probe_limits(job.input_path)
            width -= width % 2
            height -= height % 2
            cap = cv2.VideoCapture(str(job.input_path))
            raw = job.work_dir / "raw.avi"
            writer = cv2.VideoWriter(
                str(raw),
                cv2.VideoWriter_fourcc(*"MJPG"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise APIError(500, "video_encode_failed", "Could not open a video encoder.")
            detector = self._detector_factory()
            frames_out: list[dict[str, Any]] = []
            pre = inf = post = 0.0
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index >= settings.max_video_frames:
                    raise APIError(422, "too_many_frames", "Stopped because the frame cap was exceeded.")
                frame = frame[:height, :width]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                (dets,), timings = detector.detect(
                    [rgb],
                    conf_threshold=job.conf,
                    iou_threshold=job.iou,
                    max_detections=settings.max_detections,
                )
                pre += timings.preprocess_ms
                inf += timings.inference_ms
                post += timings.postprocess_ms
                writer.write(_draw(frame, dets))
                frames_out.append(
                    {
                        "index": index,
                        "time_s": round(index / fps, 3),
                        "detections": [
                            {
                                "class_id": d.class_id,
                                "label": d.label,
                                "confidence": d.confidence,
                                "box": {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2},
                            }
                            for d in dets
                        ],
                    }
                )
                index += 1
                with self._lock:
                    job.progress = index / frame_count if frame_count else 0
            if index == 0:
                raise APIError(415, "unreadable_video", "The clip had no decodable frames.")
            writer.release()
            writer = None
            if cap is not None:
                cap.release()
                cap = None
            mp4 = job.work_dir / "annotated.mp4"
            _transcode(raw, mp4)
            raw.unlink(missing_ok=True)
            pipeline_s = (pre + inf + post) / 1000
            result = {
                "frame_count": index,
                "fps": round(fps, 3),
                "width": width,
                "height": height,
                "duration_s": round(duration, 3),
                "detections_total": sum(len(frame["detections"]) for frame in frames_out),
                "latency_ms": {
                    "preprocess_total": round(pre, 3),
                    "inference_total": round(inf, 3),
                    "postprocess_total": round(post, 3),
                    "preprocess_avg": round(pre / index, 3),
                    "inference_avg": round(inf / index, 3),
                    "postprocess_avg": round(post / index, 3),
                    "pipeline_fps": round(index / pipeline_s, 3) if pipeline_s else 0,
                },
                "frames": frames_out,
            }
            with self._lock:
                job.status = "succeeded"
                job.progress = 1.0
                job.result = result
                job.video_path = mp4
                job.finished_at = time.time()
        except APIError as exc:
            self._fail(job_id, exc.body()["error"])
        except Exception:
            logger.exception("job %s failed", job_id)
            self._fail(job_id, {"code": "job_failed", "message": "The clip could not be processed."})
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            with self._lock:
                item = self._jobs.get(job_id)
                path = item.input_path if item else None
            if path is not None:
                path.unlink(missing_ok=True)

    def _fail(self, job_id: str, error: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.error = error
            job.finished_at = time.time()
            work = job.work_dir
            job.video_path = None
        shutil.rmtree(work, ignore_errors=True)

    def _sweep(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                job
                for job in self._jobs.values()
                if job.finished_at and now - job.finished_at > settings.job_ttl_seconds
            ]
            for job in expired:
                shutil.rmtree(job.work_dir, ignore_errors=True)
                del self._jobs[job.id]


_queue: JobQueue | None = None
_guard = threading.Lock()


def get_queue(detector_factory: Callable[[], Any]) -> JobQueue:
    global _queue
    with _guard:
        if _queue is None:
            _queue = JobQueue(detector_factory)
        return _queue
