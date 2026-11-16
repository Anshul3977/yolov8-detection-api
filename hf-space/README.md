---
title: YOLOv8 Detection (in-browser)
emoji: 🎯
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: agpl-3.0
short_description: YOLOv8n detection + tracking, ONNX Runtime Web in your tab
---

# YOLOv8 Detection, in your browser

YOLOv8n (COCO, 80 classes) exported to ONNX and run **client-side** with ONNX Runtime Web (WASM): images, short clips with IoU tracking, and a live webcam mode. No server is involved; frames never leave your device.

The same model is also served as a FastAPI + ONNX Runtime REST API with `/detect`, `/detect/batch`, `/detect/video`, a WebSocket stream and Prometheus metrics:

- API demo: https://yolov8-detection-api.vercel.app
- Source, benchmarks and tests: https://github.com/Anshul3977/yolov8-detection-api

Sample clip: "Queen Victoria Street (Fremantle Traffic Bridge) south", Wikimedia Commons, CC0.
