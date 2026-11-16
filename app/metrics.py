"""Prometheus metrics exposed at /metrics."""
from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "End-to-end request latency",
    ["path"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 5),
)
INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "ONNX Runtime forward-pass latency per call",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1, 2),
)
IMAGES = Counter("images_processed_total", "Images run through the model", ["endpoint"])
DETECTIONS = Counter("detections_total", "Objects returned", ["label"])
BATCH_SIZE = Histogram("batch_size", "Images per /detect/batch call", buckets=(1, 2, 4, 8, 16, 32))
MODEL_LOADED = Gauge("model_loaded", "1 when the ONNX model is loaded")
