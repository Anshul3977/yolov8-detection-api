"""Per-stage latency on the sample images, plus FPS on the sample clip.

Writes benchmarks/benchmark.json and benchmarks/summary.md. Full-precision
percentile values live in the JSON; the summary prints them to 4 decimal
places so the README table matches this file.

    python bench/stages.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IMAGES = [ROOT / "tests" / "assets" / "bus.jpg", ROOT / "tests" / "assets" / "zidane.jpg"]
CLIP = ROOT / "app" / "static" / "street.mp4"
WARMUP = 10
ITERS = 40
COMPARE_WARMUP = 5
COMPARE_ITERS = 20
CONF = 0.25
IOU = 0.45


def percentile(samples: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), q, method="linear"))


def cpu_info() -> dict:
    model = "unknown"
    mhz = None
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name") and model == "unknown":
            model = line.split(":", 1)[1].strip()
        elif line.startswith("cpu MHz") and mhz is None:
            mhz = float(line.split(":", 1)[1].strip())
    mem_kib = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal"):
            mem_kib = int(line.split()[1])
            break
    return {
        "model_name": model,
        "cpu_mhz": mhz,
        "logical_cpus": os.cpu_count(),
        "mem_total_kib": mem_kib,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_once(det, image: np.ndarray) -> tuple[float, float, float]:
    t0 = time.perf_counter()
    tensor, letterboxes = det.preprocess([image])
    t1 = time.perf_counter()
    output = det.forward(tensor)
    t2 = time.perf_counter()
    det.postprocess(output, letterboxes, [image.shape[:2]], CONF, IOU, 300)
    t3 = time.perf_counter()
    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000


def measure_images(det, images: list[tuple[np.ndarray, Path]], warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        for image, _path in images:
            stage_once(det, image)
    per_image = {}
    pooled = {"preprocess_ms": [], "inference_ms": [], "postprocess_ms": []}
    for image, path in images:
        pre, inf, post = [], [], []
        for _ in range(iters):
            a, b, c = stage_once(det, image)
            pre.append(a)
            inf.append(b)
            post.append(c)
        per_image[path.name] = {"preprocess_ms": pre, "inference_ms": inf, "postprocess_ms": post}
        for key, values in (("preprocess_ms", pre), ("inference_ms", inf), ("postprocess_ms", post)):
            pooled[key].extend(values)
    return {"per_image": per_image, "pooled": pooled}


def stats(samples: list[float]) -> dict:
    return {"p50_ms": percentile(samples, 50), "p95_ms": percentile(samples, 95), "n": len(samples)}


def load_images():
    import cv2

    loaded = []
    for path in IMAGES:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise SystemExit(f"missing sample image {path}")
        loaded.append((cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), path))
    return loaded


def measure_clip(det) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(CLIP))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames in {CLIP}")
    for frame in frames[:5]:
        stage_once(det, frame)
    pre, inf, post = [], [], []
    wall0 = time.perf_counter()
    for frame in frames:
        a, b, c = stage_once(det, frame)
        pre.append(a)
        inf.append(b)
        post.append(c)
    wall = time.perf_counter() - wall0
    stage_s = (sum(pre) + sum(inf) + sum(post)) / 1000
    return {
        "file": "app/static/street.mp4",
        "frames": len(frames),
        "source_fps": fps,
        "preprocess_ms": stats(pre),
        "inference_ms": stats(inf),
        "postprocess_ms": stats(post),
        "pipeline_fps": len(frames) / stage_s,
        "wall_fps": len(frames) / wall,
        "wall_s": wall,
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def summary_markdown(payload: dict) -> str:
    host = payload["host"]
    primary = payload["runs"]["threads_primary"]
    lines = [
        f"Host `{host['model_name']}`, {host['logical_cpus']} logical CPUs"
        + ("" if host["cpu_mhz"] is None else f" at {host['cpu_mhz']} MHz")
        + f", MemTotal {host['mem_total_kib']} kB. "
        f"ONNX Runtime {payload['runtime']['onnxruntime']}, "
        f"intra-op threads {primary['intra_op_threads']}, "
        f"input {payload['model']['input_shape']}. "
        f"Model sha256 `{payload['model']['sha256'][:16]}…`. "
        f"{primary['warmup_iterations']} warmup iterations then {primary['measured_iterations']} measured "
        f"iterations per image. Percentiles use NumPy's linear method. "
        f"Values below are shown to 4 decimal places; exact floats are in `benchmarks/benchmark.json`.",
        "",
        "| Image | Pre p50 | Pre p95 | Infer p50 | Infer p95 | Post p50 | Post p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = dict(primary["per_image"])
    rows["all images"] = primary["pooled"]
    for name, row in rows.items():
        lines.append(
            "| {name} | {pre50} | {pre95} | {inf50} | {inf95} | {post50} | {post95} |".format(
                name=name,
                pre50=fmt(row["preprocess_ms"]["p50_ms"]),
                pre95=fmt(row["preprocess_ms"]["p95_ms"]),
                inf50=fmt(row["inference_ms"]["p50_ms"]),
                inf95=fmt(row["inference_ms"]["p95_ms"]),
                post50=fmt(row["postprocess_ms"]["p50_ms"]),
                post95=fmt(row["postprocess_ms"]["p95_ms"]),
            )
        )
    clip = primary["clip"]
    one = payload["runs"]["threads_1"]
    lines.extend(
        [
            "",
            f"Sample clip `{clip['file']}`: {clip['frames']} frames, source {clip['source_fps']} fps. "
            f"Pipeline FPS (frames / sum of preprocess+inference+postprocess) **{fmt(clip['pipeline_fps'])}**. "
            f"Wall-clock FPS of that same loop **{fmt(clip['wall_fps'])}**.",
            "",
            "| Run | Infer p50 | Infer p95 | Clip pipeline FPS |",
            "|---|---:|---:|---:|",
            f"| {primary['intra_op_threads']} threads, 640 | {fmt(primary['pooled']['inference_ms']['p50_ms'])} | {fmt(primary['pooled']['inference_ms']['p95_ms'])} | {fmt(clip['pipeline_fps'])} |",
            f"| 1 thread, 640 | {fmt(one['pooled']['inference_ms']['p50_ms'])} | {fmt(one['pooled']['inference_ms']['p95_ms'])} | — |",
            "",
        ]
    )
    return "\n".join(lines)


def reduce_run(measured: dict, clip: dict | None, threads: int, warmup: int, iters: int) -> dict:
    per_image = {}
    for name, raw in measured["per_image"].items():
        per_image[name] = {key: stats(values) for key, values in raw.items()}
    pooled = {key: stats(values) for key, values in measured["pooled"].items()}
    body = {
        "intra_op_threads": threads,
        "warmup_iterations": warmup,
        "measured_iterations": iters,
        "per_image": per_image,
        "pooled": pooled,
    }
    if clip is not None:
        body["clip"] = clip
    return body


def main() -> None:
    import cv2
    import onnxruntime as ort
    from app.detector import Detector

    loaded = load_images()
    images = [(image, path) for image, path in loaded]
    threads = os.cpu_count() or 1
    primary_det = Detector(ROOT / "models" / "yolov8n.onnx", threads=threads)
    input_shape = primary_det.session.get_inputs()[0].shape
    measured = measure_images(primary_det, images, WARMUP, ITERS)
    clip = measure_clip(primary_det)
    one = Detector(ROOT / "models" / "yolov8n.onnx", threads=1)
    measured_one = measure_images(one, images, COMPARE_WARMUP, COMPARE_ITERS)

    payload = {
        "host": cpu_info(),
        "runtime": {
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "provider": "CPUExecutionProvider",
        },
        "model": {
            "file": "models/yolov8n.onnx",
            "sha256": sha256(ROOT / "models" / "yolov8n.onnx"),
            "input_shape": input_shape,
            "conf": CONF,
            "iou": IOU,
        },
        "protocol": {
            "percentile": "numpy.percentile method=linear",
            "images": [path.name for _, path in images],
            "clip": "app/static/street.mp4",
            "note": "Stage times are perf_counter around preprocess, session.run, and postprocess. Drawing and JPEG encode are not included.",
        },
        "runs": {
            "threads_primary": reduce_run(measured, clip, threads, WARMUP, ITERS),
            "threads_1": reduce_run(measured_one, None, 1, COMPARE_WARMUP, COMPARE_ITERS),
        },
    }
    out_dir = ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")
    text = summary_markdown(payload)
    (out_dir / "summary.md").write_text(text)
    readme = ROOT / "README.md"
    start, end = "<!-- BENCHMARK_STAGES:START -->", "<!-- BENCHMARK_STAGES:END -->"
    body = readme.read_text()
    if start in body and end in body:
        pre, rest = body.split(start, 1)
        _, post = rest.split(end, 1)
        readme.write_text(f"{pre}{start}\n{text}{end}{post}")
    print(text)


if __name__ == "__main__":
    main()
