import time

import cv2
import numpy as np


def _clip(path, frames: int, fps: float, size: tuple[int, int] = (160, 120)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    for i in range(frames):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        frame[:] = (30, 60, 90)
        cv2.rectangle(frame, (10 + i, 10), (80, 90), (0, 220, 0), -1)
        writer.write(frame)
    writer.release()


def _wait(client, job_id: str) -> dict:
    deadline = time.time() + 30
    body = {}
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in {"succeeded", "failed"}:
            return body
        time.sleep(0.2)
    raise AssertionError(f"job timed out: {body}")


def test_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_job_lifecycle_on_tiny_clip(client, tmp_path):
    path = tmp_path / "tiny.avi"
    _clip(path, frames=3, fps=10)
    with path.open("rb") as handle:
        response = client.post(
            "/jobs",
            files={"file": ("tiny.avi", handle, "video/x-msvideo")},
            data={"conf": "0.25", "iou": "0.45"},
        )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    body = _wait(client, job_id)
    assert body["status"] == "succeeded", body
    assert body["result"]["frame_count"] == 3
    assert len(body["result"]["frames"]) == 3
    assert "preprocess_avg" in body["result"]["latency_ms"]
    video = client.get(f"/jobs/{job_id}/video")
    assert video.status_code == 200
    assert video.headers["content-type"].startswith("video/mp4")
    assert len(video.content) > 100


def test_job_rejects_long_clip(client, tmp_path):
    path = tmp_path / "long.avi"
    _clip(path, frames=11, fps=1, size=(64, 48))
    with path.open("rb") as handle:
        response = client.post(
            "/jobs",
            files={"file": ("long.avi", handle, "video/x-msvideo")},
            data={"conf": "0.25"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "video_too_long"


def test_job_rejects_bad_confidence(client, tmp_path):
    path = tmp_path / "tiny.avi"
    _clip(path, frames=2, fps=10, size=(64, 48))
    with path.open("rb") as handle:
        response = client.post(
            "/jobs",
            files={"file": ("tiny.avi", handle, "video/x-msvideo")},
            data={"conf": "1.5"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_conf"


def test_job_rejects_text(client):
    response = client.post("/jobs", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media"
