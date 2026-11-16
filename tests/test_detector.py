import numpy as np

from app.detector import letterbox, nms


def test_letterbox_keeps_aspect_ratio_and_pads():
    img = np.zeros((480, 640, 3), np.uint8)
    out, lb = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert lb.scale == 1.0
    assert (lb.pad_x, lb.pad_y) == (0, 80)
    assert (out[:80] == 114).all() and (out[80:560] == 0).all()


def test_letterbox_downscales_large_images():
    out, lb = letterbox(np.zeros((1080, 810, 3), np.uint8), 640)
    assert out.shape == (640, 640, 3)
    assert abs(lb.scale - 640 / 1080) < 1e-9


def test_nms_suppresses_overlaps_and_keeps_best_first():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.6, 0.9, 0.5], dtype=np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [1, 2]


def test_nms_keeps_everything_when_disjoint():
    boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float32)
    assert sorted(nms(boxes, np.array([0.5, 0.4]), 0.5).tolist()) == [0, 1]
