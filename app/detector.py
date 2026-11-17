"""YOLOv8 inference on ONNX Runtime: letterbox -> forward -> decode -> NMS.

No PyTorch at serving time; the only heavy dependency is onnxruntime.
"""
from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Timings:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float


def letterbox(image: np.ndarray, size: int = 640, fill: int = 114) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio, pad to size x size. image is HxWx3 uint8 RGB."""
    import cv2  # local import keeps module import cheap for tooling

    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if (new_w, new_h) != (w, h) else image
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, Letterbox(scale=scale, pad_x=left, pad_y=top)


def scale_xyxy(boxes: np.ndarray, lb: Letterbox, width: int, height: int) -> np.ndarray:
    """Map xyxy boxes from letterboxed pixels back onto the original image."""
    out = boxes.copy()
    out[:, [0, 2]] = ((out[:, [0, 2]] - lb.pad_x) / lb.scale).clip(0, width)
    out[:, [1, 3]] = ((out[:, [1, 3]] - lb.pad_y) / lb.scale).clip(0, height)
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression. boxes: (N, 4) xyxy. Returns kept indices, best first."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


class Detector:
    def __init__(self, model_path: str | Path, imgsz: int = 640, threads: int = 0) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.model_name = Path(model_path).stem

    def preprocess(self, images: list[np.ndarray]) -> tuple[np.ndarray, list[Letterbox]]:
        batch, boxes = [], []
        for img in images:
            padded, lb = letterbox(img, self.imgsz)
            batch.append(padded)
            boxes.append(lb)
        tensor = np.stack(batch).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor), boxes

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: tensor})[0]

    def postprocess(
        self,
        output: np.ndarray,
        letterboxes: list[Letterbox],
        shapes: list[tuple[int, int]],
        conf_threshold: float,
        iou_threshold: float,
        max_detections: int,
    ) -> list[list[Detection]]:
        results: list[list[Detection]] = []
        for pred, lb, (h, w) in zip(output, letterboxes, shapes):
            pred = pred.T  # (anchors, 4 + num_classes)
            class_scores = pred[:, 4:]
            class_ids = class_scores.argmax(axis=1)
            scores = class_scores[np.arange(len(class_ids)), class_ids]
            mask = scores >= conf_threshold
            if not mask.any():
                results.append([])
                continue
            cxcywh, scores, class_ids = pred[mask, :4], scores[mask], class_ids[mask]
            xyxy = np.empty_like(cxcywh)
            xyxy[:, :2] = cxcywh[:, :2] - cxcywh[:, 2:] / 2
            xyxy[:, 2:] = cxcywh[:, :2] + cxcywh[:, 2:] / 2
            # Class-aware NMS: shift each class into its own coordinate range.
            keep = nms(xyxy + class_ids[:, None] * 7680.0, scores, iou_threshold)[:max_detections]
            xyxy = scale_xyxy(xyxy[keep], lb, w, h)
            results.append(
                [
                    Detection(
                        class_id=int(c),
                        label=self.names.get(int(c), str(int(c))),
                        confidence=round(float(s), 4),
                        x1=round(float(b[0]), 1),
                        y1=round(float(b[1]), 1),
                        x2=round(float(b[2]), 1),
                        y2=round(float(b[3]), 1),
                    )
                    for b, s, c in zip(xyxy, scores[keep], class_ids[keep])
                ]
            )
        return results

    def detect(
        self,
        images: list[np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
    ) -> tuple[list[list[Detection]], Timings]:
        t0 = time.perf_counter()
        tensor, letterboxes = self.preprocess(images)
        t1 = time.perf_counter()
        output = self.forward(tensor)
        t2 = time.perf_counter()
        shapes = [img.shape[:2] for img in images]
        detections = self.postprocess(output, letterboxes, shapes, conf_threshold, iou_threshold, max_detections)
        t3 = time.perf_counter()
        return detections, Timings(
            preprocess_ms=round((t1 - t0) * 1000, 2),
            inference_ms=round((t2 - t1) * 1000, 2),
            postprocess_ms=round((t3 - t2) * 1000, 2),
        )
