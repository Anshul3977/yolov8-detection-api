from pathlib import Path

import cv2
import numpy as np
import pytest
from app.detector import Detection
from app.tracker import IoUTracker, iou_matrix

ASSETS = Path(__file__).parent / "assets"


def _det(x1, y1, x2, y2, cls=0, label="person", conf=0.9):
    return Detection(cls, label, conf, x1, y1, x2, y2)


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-4)


def test_tracker_keeps_id_while_object_moves():
    t = IoUTracker(min_hits=2)
    ids = [t.update([_det(10 + 3 * i, 10, 60 + 3 * i, 110)])[0].track_id for i in range(6)]
    assert len(set(ids)) == 1
    assert t.summary() == {"person": 1}


def test_tracker_separates_objects_and_classes():
    t = IoUTracker(min_hits=1)
    for i in range(3):
        out = t.update([_det(0, 0, 50, 50), _det(200, 0, 250, 50), _det(0, 0, 50, 50, cls=2, label="car")])
    assert len({o.track_id for o in out}) == 3
    assert t.summary() == {"person": 2, "car": 1}


def test_tracker_drops_stale_tracks_and_ignores_flicker():
    t = IoUTracker(max_age=1, min_hits=2)
    t.update([_det(0, 0, 50, 50)])  # seen once: never confirmed
    for _ in range(3):
        t.update([])
    assert t.tracks == [] and t.summary() == {}
    first = t.update([_det(0, 0, 50, 50)])[0].track_id
    assert first == 2  # old id was not reused


@pytest.fixture(scope="module")
def pan_clip(tmp_path_factory) -> bytes:
    """2 s, 15 fps clip that slowly pans across bus.jpg, so objects move between frames."""
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    h, w = img.shape[:2]
    cw, ch = 640, 640
    path = tmp_path_factory.mktemp("v") / "pan.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (cw, ch))
    for i in range(30):
        x = int((w - cw) * i / 29)
        out.write(cv2.resize(img[:, x : x + cw] if w > cw else img, (cw, ch)))
    out.release()
    return path.read_bytes()


def test_detect_video_tracks_objects(client, pan_clip):
    r = client.post("/detect/video?sample_fps=5", files={"file": ("pan.mp4", pan_clip, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["video"]["fps"] == pytest.approx(15, abs=0.5)
    assert body["frame_step"] == 3 and body["frames_analysed"] == 10
    assert body["tracking"] is True
    assert body["unique_objects"].get("person", 0) >= 2
    first = body["frames"][0]["detections"]
    assert first and all("track_id" in d for d in first)
    assert body["timings"]["processing_fps"] > 0


def test_detect_video_without_tracking(client, pan_clip):
    r = client.post("/detect/video?track=false&max_frames=3", files={"file": ("pan.mp4", pan_clip, "video/mp4")})
    assert r.status_code == 200
    body = r.json()
    assert body["frames_analysed"] == 3 and body["unique_objects"] is None
    assert "track_id" not in body["frames"][0]["detections"][0]


def test_detect_video_rejects_non_video(client):
    r = client.post("/detect/video", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_websocket_stream(client):
    frame = (ASSETS / "bus.jpg").read_bytes()
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_bytes(frame)
        a = ws.receive_json()
        ws.send_bytes(frame)
        b = ws.receive_json()
        ws.send_bytes(b"garbage")
        err = ws.receive_json()
    assert any(d["label"] == "bus" for d in a["detections"])
    # same frame twice -> same track ids
    assert sorted(d["track_id"] for d in a["detections"]) == sorted(d["track_id"] for d in b["detections"])
    assert "error" in err
