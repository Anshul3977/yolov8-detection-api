"""Lightweight multi-object tracker (IoU association, SORT-style without a Kalman filter).

Each frame, detections are matched to existing tracks of the same class by
greedy highest-IoU-first assignment. Unmatched detections start new tracks;
tracks unseen for more than `max_age` frames are dropped. A track is only
"confirmed" (counted) after `min_hits` matches, which filters one-frame flickers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.detector import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / (area_a + area_b - inter + 1e-9)


@dataclass
class Track:
    track_id: int
    class_id: int
    label: str
    box: np.ndarray
    hits: int = 1
    age: int = 0  # frames since last match
    first_frame: int = 0
    last_frame: int = 0


@dataclass
class TrackedDetection:
    track_id: int
    detection: Detection


@dataclass
class IoUTracker:
    iou_threshold: float = 0.3
    max_age: int = 5
    min_hits: int = 2
    tracks: list[Track] = field(default_factory=list)
    _next_id: int = 1
    _frame: int = 0
    _confirmed: dict[int, Track] = field(default_factory=dict)

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        frame = self._frame
        self._frame += 1
        det_boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32).reshape(-1, 4)
        trk_boxes = np.array([t.box for t in self.tracks], dtype=np.float32).reshape(-1, 4)
        ious = iou_matrix(trk_boxes, det_boxes)
        # Only same-class pairs may match.
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(detections):
                if t.class_id != d.class_id:
                    ious[ti, di] = 0.0

        assigned_det: dict[int, Track] = {}
        used_trk: set[int] = set()
        if ious.size:
            for flat in np.argsort(-ious, axis=None):
                ti, di = divmod(int(flat), ious.shape[1])
                if ious[ti, di] < self.iou_threshold:
                    break
                if ti in used_trk or di in assigned_det:
                    continue
                used_trk.add(ti)
                assigned_det[di] = self.tracks[ti]

        for ti, t in enumerate(self.tracks):
            if ti not in used_trk:
                t.age += 1

        out: list[TrackedDetection] = []
        for di, d in enumerate(detections):
            t = assigned_det.get(di)
            if t is None:
                t = Track(self._next_id, d.class_id, d.label, det_boxes[di], first_frame=frame, last_frame=frame)
                self._next_id += 1
                self.tracks.append(t)
            else:
                t.box, t.hits, t.age, t.last_frame = det_boxes[di], t.hits + 1, 0, frame
            if t.hits >= self.min_hits:
                self._confirmed[t.track_id] = t
            out.append(TrackedDetection(t.track_id, d))

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return out

    def summary(self) -> dict[str, int]:
        """Unique confirmed objects per class seen so far."""
        counts: dict[str, int] = {}
        for t in self._confirmed.values():
            counts[t.label] = counts.get(t.label, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    @property
    def confirmed_tracks(self) -> list[Track]:
        return list(self._confirmed.values())
