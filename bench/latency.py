"""PyTorch vs ONNX Runtime latency for YOLOv8n on CPU.

Both backends get the identical preprocessed 1x3x640x640 tensor and the same thread
count, so the comparison isolates the forward pass. A second section times the full
ONNX pipeline (letterbox + forward + decode + NMS) on a real image.

    python bench/latency.py --threads 4 --iters 200 --out bench/results/latency.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def summarize(samples_ms: list[float]) -> dict:
    arr = np.asarray(samples_ms)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "fps": round(1000.0 / float(arr.mean()), 1),
        "n": len(samples_ms),
    }


def timeit(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--onnx", default=str(ROOT / "models" / "yolov8n.onnx"))
    p.add_argument("--image", default=str(ROOT / "tests" / "assets" / "bus.jpg"))
    p.add_argument("--out")
    a = p.parse_args()

    import onnxruntime as ort
    import torch
    from app.detector import Detector
    from PIL import Image
    from ultralytics import YOLO

    torch.set_num_threads(a.threads)
    image = np.asarray(Image.open(a.image).convert("RGB"))
    det = Detector(a.onnx, threads=a.threads)
    tensor, _ = det.preprocess([image])

    torch_model = YOLO(a.weights).model.fuse().eval()
    x = torch.from_numpy(tensor)

    @torch.inference_mode()
    def torch_forward():
        torch_model(x)

    results = {
        "env": {
            "cpu": platform.processor() or platform.machine(),
            "logical_cpus": os.cpu_count(),
            "threads": a.threads,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "input": list(tensor.shape),
        }
    }
    results["pytorch_forward"] = summarize(timeit(torch_forward, a.warmup, a.iters))
    results["onnxruntime_forward"] = summarize(timeit(lambda: det.forward(tensor), a.warmup, a.iters))
    results["onnxruntime_end_to_end"] = summarize(timeit(lambda: det.detect([image]), a.warmup, a.iters))

    # Parity check: same raw output from both backends.
    with torch.inference_mode():
        t_out = torch_model(x)[0].numpy()
    o_out = det.forward(tensor)
    results["max_abs_diff_raw_output"] = float(np.abs(t_out - o_out).max())
    results["speedup_p50"] = round(results["pytorch_forward"]["p50_ms"] / results["onnxruntime_forward"]["p50_ms"], 2)

    print(json.dumps(results, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
