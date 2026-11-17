# YOLOv8 Detection API

[![CI](https://github.com/Anshul3977/yolov8-detection-api/actions/workflows/ci.yml/badge.svg)](https://github.com/Anshul3977/yolov8-detection-api/actions/workflows/ci.yml)

**Live demos**
- API + UI on Vercel: https://yolov8-detection-api.vercel.app (image, video + tracking, and webcam modes; [OpenAPI docs](https://yolov8-detection-api.vercel.app/docs))
- In-browser version on Hugging Face Spaces: https://huggingface.co/spaces/ANSHUL3977/yolov8-detection-demo. The same ONNX model runs client-side with ONNX Runtime Web (WASM), with a live webcam tracker and no server round trip.

![Video mode: cars tracked across frames with persistent IDs](docs/demo-video-tracking.png)

A production-style object detection service. Pretrained **YOLOv8n** (COCO, 80 classes) is exported to **ONNX** and served with **ONNX Runtime** behind **FastAPI**. PyTorch is only used to export the model and to benchmark against, so the serving image doesn't carry it.

- `POST /detect`: one image in, boxes, labels and confidences out
- `POST /detect/batch`: up to 16 images in **one** forward pass (dynamic batch axis)
- `POST /detect/video`: a short clip in, frames sampled at `sample_fps`, batched through the model, and linked into **tracks** so each object is counted once (`unique_objects`)
- `WS /ws/detect`: streaming mode for webcams. Send JPEG frames, get detections with persistent track IDs back per frame (needs a long-lived server, so it runs in Docker but not on Vercel)
- `GET /health`: model and runtime status
- `GET /ready`: 200 only after the ONNX session has loaded
- `POST /jobs`: queue a short clip (≤ 10 s). One job runs at a time. Poll `GET /jobs/{id}` for status, progress and per-frame detections, then `GET /jobs/{id}/video` for the annotated MP4
- `GET /metrics`: Prometheus metrics (request rate, latency histograms, inference time, detections per class)
- `GET /docs`: OpenAPI UI, and `GET /` is a small browser demo that draws the boxes

## Architecture

```mermaid
flowchart LR
    C[Client / browser demo] -->|multipart image| A[FastAPI]
    A --> V[Validate and decode<br/>size limit, EXIF, RGB]
    V --> P[Letterbox 640x640<br/>normalize, NCHW]
    P --> O[ONNX Runtime<br/>YOLOv8n, CPU]
    O --> N[Decode + class-aware NMS<br/>NumPy]
    N --> A
    A -->|JSON boxes + timings| C
    N --> T[IoU tracker<br/>video and WebSocket only]
    T --> A
    A -.-> M[(Prometheus /metrics)]
    subgraph Offline
      W[yolov8n.pt] -->|scripts/export_onnx.py| X[models/yolov8n.onnx]
    end
    X --> O
```

Pre- and post-processing are my own NumPy code (letterbox, box decoding, NMS), not the Ultralytics predictor. The raw ONNX output matches PyTorch to within 0.0014 (max absolute difference over the 1×84×8400 output).

## Benchmarks

All numbers were measured on CPU: an 8 vCPU Intel Xeon cloud VM, Python 3.11, torch 2.5.1+cpu, onnxruntime 1.32.0, 640×640 input, 200 timed runs after 20 warm-up runs. The VM was shared with other jobs, so treat the tail numbers (p95) as conservative. Raw JSON is in [`bench/results/`](bench/results/).

### Per-stage latency

<!-- BENCHMARK_STAGES:START -->
Host `Intel(R) Xeon(R) Processor`, 4 logical CPUs at 2400.0 MHz, MemTotal 16387328 kB. ONNX Runtime 1.32.0, intra-op threads 4, input ['batch', 3, 'height', 'width']. Model sha256 `c7e13e3fe57a28b5…`. 10 warmup iterations then 40 measured iterations per image. Percentiles use NumPy's linear method. Values below are shown to 4 decimal places; exact floats are in `benchmarks/benchmark.json`. Every iteration letterboxes the photo to 640×640 before the forward pass.

| Image | Pre p50 | Pre p95 | Infer p50 | Infer p95 | Post p50 | Post p95 |
|---|---:|---:|---:|---:|---:|---:|
| bus.jpg | 1.6942 | 1.7658 | 19.2595 | 20.1809 | 0.7563 | 0.8598 |
| zidane.jpg | 1.3753 | 1.4832 | 19.2655 | 19.5779 | 0.6837 | 0.7195 |
| all images | 1.5304 | 1.7517 | 19.2647 | 20.1619 | 0.7349 | 0.7941 |

Sample clip `app/static/street.mp4`: 120 frames, source 15.0 fps. Pipeline FPS (frames / sum of preprocess+inference+postprocess) **42.2275**. Wall-clock FPS of that same loop **41.9929**.

| Run | Infer p50 | Infer p95 | Clip pipeline FPS |
|---|---:|---:|---:|
| 4 threads, 640 | 19.2647 | 20.1619 | 42.2275 |
| 1 thread, 640 | 50.5584 | 50.8580 | — |
<!-- BENCHMARK_STAGES:END -->

### Forward pass: PyTorch vs ONNX Runtime

Both backends get the identical preprocessed tensor and the same thread count.

| Threads | Backend | p50 | p95 | Throughput |
|---|---|---|---|---|
| 1 | PyTorch | 69.1 ms | 77.5 ms | 14.3 fps |
| 1 | **ONNX Runtime** | **51.7 ms** | **56.7 ms** | **19.0 fps** |
| 4 | PyTorch | 31.8 ms | 36.5 ms | 31.0 fps |
| 4 | **ONNX Runtime** | **19.9 ms** | **22.4 ms** | **49.0 fps** |

ONNX Runtime is **1.34× faster at p50 on 1 thread and 1.6× faster on 4 threads**.

Full pipeline on a real 810×1080 photo (decode excluded; letterbox + forward + NMS), ONNX Runtime: p50 54.3 ms on 1 thread, p50 27.2 ms on 4 threads.

### Load test

`bench/loadtest.py` keeps N clients posting `bus.jpg` to `/detect` for 20 s against one uvicorn worker (`ORT_THREADS=4`), with the load generator pinned to separate cores.

| Concurrent clients | Requests/s | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|
| 1 | 28.9 | 32.3 ms | 41.0 ms | 79.1 ms | 0 |
| 4 | 80.3 | 47.7 ms | 63.6 ms | 114.5 ms | 0 |
| 8 | **90.1** | 84.5 ms | 127.6 ms | 166.5 ms | 0 |

These are end-to-end HTTP numbers, including multipart upload, JPEG decode and JSON.

Reproduce:

```bash
pip install -r requirements-bench.txt --extra-index-url https://download.pytorch.org/whl/cpu
python bench/latency.py --threads 4 --iters 200
uvicorn app.main:app --port 8000 &
python bench/loadtest.py --url http://127.0.0.1:8000 --concurrency 8 --duration 20
```

### Hosted demo (Vercel serverless, CPU)

Measured from a client in India against the production deployment, 50 sequential `POST /detect` calls with `bus.jpg` after 3 warm-up calls:

| | p50 | p95 |
|---|---|---|
| Server-side model inference | 170.3 ms | 193.8 ms |
| Full round trip (upload + network + JSON) | 1.54 s | 1.65 s |

On the 8 s sample street clip (`/detect/video`, 40 frames sampled at 5 fps), the hosted function processed 4.96 frames/s and counted 10 unique cars and 3 trucks. Serverless CPUs are much slower than the 8 vCPU VM above, so this is the floor, not the ceiling.

## Tracking

`app/tracker.py` is a SORT-style tracker without the Kalman filter. Each new frame's detections are matched to live tracks of the same class, highest IoU first (threshold 0.3). Unmatched detections open new tracks, and tracks unseen for more than 5 frames are dropped. A track only counts toward `unique_objects` after 2 matches, which filters out one-frame flicker. It's deliberately simple. Fast motion at low `sample_fps` breaks IoU overlap, and a Kalman motion model or ByteTrack-style low-score matching would be the next step.

```bash
curl -F "file=@app/static/street.mp4" "localhost:8000/detect/video?sample_fps=5"
# {"frames_analysed": 40, "unique_objects": {"car": 10, "truck": 3, ...},
#  "frames": [{"frame": 0, "t": 0.0, "detections": [{"track_id": 1, "label": "car", ...}]}, ...]}
```

WebSocket (Python client):

```python
import asyncio, websockets
async def main():
    async with websockets.connect("ws://localhost:8000/ws/detect") as ws:
        await ws.send(open("frame.jpg", "rb").read())
        print(await ws.recv())  # {"frame": 0, "detections": [{"track_id": 1, ...}], "unique_objects": {...}}
asyncio.run(main())
```

## Quick start

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000
curl -F "file=@tests/assets/bus.jpg" "localhost:8000/detect?conf=0.25"
```

```json
{
  "model": "yolov8n",
  "filename": "bus.jpg",
  "image": {"width": 810, "height": 1080},
  "count": 5,
  "detections": [
    {"class_id": 0, "label": "person", "confidence": 0.8902, "box": {"x1": 670.4, "y1": 380.6, "x2": 809.9, "y2": 879.7}},
    {"class_id": 5, "label": "bus", "confidence": 0.8434, "box": {"x1": 31.5, "y1": 230.7, "x2": 801.4, "y2": 775.8}}
  ],
  "timings": {"preprocess_ms": 2.51, "inference_ms": 16.62, "postprocess_ms": 0.94}
}
```

(Output trimmed to two detections.)

Batch:

```bash
curl -F "files=@tests/assets/bus.jpg" -F "files=@tests/assets/zidane.jpg" localhost:8000/detect/batch
```

### Docker

```bash
docker compose up --build
```

The image installs only `requirements.txt` (ONNX Runtime, no PyTorch) plus `ffmpeg` for annotated clips. It runs as uid 1000 and listens on `$PORT` (7860), so the same image is a Hugging Face Docker Space. `models/yolov8n.onnx` is the file produced by `scripts/export_onnx.py` from the official Ultralytics `yolov8n.pt` weights and copied into the image.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `models/yolov8n.onnx` | ONNX model file |
| `CONF_THRESHOLD` | `0.25` | Default confidence threshold (override per request with `?conf=`) |
| `IOU_THRESHOLD` | `0.45` | NMS IoU threshold (`?iou=`) |
| `MAX_DETECTIONS` | `300` | Max boxes per image (`?max_det=`) |
| `MAX_BATCH` | `16` | Max images per `/detect/batch` call |
| `MAX_UPLOAD_MB` | `10` | Per-file upload limit (413 above it) |
| `MAX_VIDEO_SECONDS` | `10` | Clip duration cap for `/jobs` |
| `MAX_VIDEO_FRAMES` | `300` | Frame cap for `/jobs` |
| `ORT_THREADS` | `0` (auto) | ONNX Runtime intra-op threads |
| `PORT`, `WORKERS` | `7860`, `1` | Container only |

Errors: 415 for files that aren't images, 413 for files over the size limit or batches that are too large, 422 for bad query parameters.

## Tests and CI

```bash
pip install -r requirements-dev.txt
pytest -v
```

21 tests cover letterbox and NMS, the tracker (ID persistence, class separation, track expiry), video upload with tracking on a generated clip, the WebSocket stream, real detections on sample images (a bus and 3+ people in `bus.jpg`), batch-vs-single consistency, threshold filtering, error codes, and the metrics output. GitHub Actions runs the tests, then builds the Docker image and smoke-tests `/health`, `/detect` and `/metrics` against the running container.

## Project layout

```
app/            FastAPI app, ONNX detector, IoU tracker, Prometheus metrics, demo page
index.py        Vercel entrypoint
models/         yolov8n.onnx (exported with dynamic batch axis, opset 17)
scripts/        export_onnx.py, deploy_hf_space.py
bench/          latency.py, loadtest.py, results/*.json
tests/          pytest suite + sample images
hf-space/       in-browser demo (ONNX Runtime Web) published to Hugging Face Spaces
```

## Deploy

**Vercel** (current live demo): `index.py` at the repo root exposes the FastAPI app, so `vercel deploy --prod` is all it takes. `vercel.json` gives the function 2 GB of memory and bundles `models/`.

**Hugging Face Spaces**

The in-browser demo is a static Space (`hf-space/`, ONNX Runtime Web). Publish it with:

```bash
HF_TOKEN=<write token> python scripts/deploy_hf_space.py --space <hf-user>/yolov8-detection-demo
```

The API itself is also Docker-Space ready. Free CPU basic hardware sleeps after inactivity; the first request after sleep pays a cold start while the ONNX session loads. One-time setup:

1. Create a Space named `yolov8-detection-api`, SDK **Docker**, hardware **CPU basic**, public.
2. Create a Hugging Face token that can write to that Space.
3. In this GitHub repo, Settings → Secrets and variables → Actions:
   - Secret `HF_TOKEN` = that token
   - Variable `HF_SPACE` = `<huggingface-username>/yolov8-detection-api`
4. Re-run **Sync Hugging Face Space**, or push to `main`. The workflow uploads the repo and prepends this header to the Space README only:

```yaml
---
title: YOLOv8 Detection API
emoji: "🎯"
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
---
```

If `HF_TOKEN` or `HF_SPACE` is unset, the workflow exits successfully and does nothing. The ONNX file is about 13 MB, so the Hub stores it with Git LFS (the upload uses `huggingface_hub`, which does that). The first Space build takes a few minutes. The Space URL is `https://huggingface.co/spaces/<huggingface-username>/yolov8-detection-api`.

## Limitations

- CPU only. A 640 input is tens of milliseconds per frame on a small VM, not a realtime multi-camera service.
- Clip jobs are in-memory, one at a time, and disappear when the process restarts or after `JOB_TTL_SECONDS` (default 30 minutes).
- Clips are capped at 10 seconds, 300 frames and the upload size limit. Longer video is rejected with a structured error.
- A free Hugging Face Space sleeps when idle. Waking it reloads the model.
- YOLOv8 weights are AGPL-3.0. Serving this API on a network is a reason to keep the source public.

## License

YOLOv8 weights and the Ultralytics export tooling are AGPL-3.0, so this repository is AGPL-3.0 too (`LICENSE`). The sample images in `tests/assets` and `app/static/bus.jpg` come from the Ultralytics repository. The demo clip `app/static/street.mp4` is cut from "Queen Victoria Street (Fremantle Traffic Bridge) south" on Wikimedia Commons (CC0).

![Annotated bus sample](docs/screenshots/ui-annotated.png)
