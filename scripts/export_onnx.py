"""Export the pretrained YOLOv8n checkpoint to ONNX.

Usage:
    pip install -r requirements-export.txt
    python scripts/export_onnx.py            # writes models/yolov8n.onnx
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out", default="models/yolov8n.onnx")
    args = parser.parse_args()

    model = YOLO(args.weights)
    # dynamic=True keeps the batch axis dynamic so /detect/batch runs one forward pass.
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, dynamic=True, simplify=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(exported, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
