---
title: YOLOv8 Detection API
emoji: 🎯
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
short_description: YOLOv8n object detection API on ONNX Runtime (CPU)
---

# YOLOv8 Detection API

FastAPI + ONNX Runtime service for pretrained YOLOv8n (COCO, 80 classes).

- `POST /detect`: one image, returns boxes, labels and confidences
- `POST /detect/batch`: several images in one forward pass
- `GET /health`, `GET /metrics` (Prometheus), `GET /docs` (OpenAPI)

Source, benchmarks and tests: https://github.com/Anshul3977/yolov8-detection-api
