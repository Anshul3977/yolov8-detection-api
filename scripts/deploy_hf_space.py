"""Publish the in-browser demo (hf-space/) to a Hugging Face *static* Space.

    pip install huggingface_hub
    HF_TOKEN=hf_xxx python scripts/deploy_hf_space.py --space <hf-username>/yolov8-detection-demo

Static Spaces are free. The page runs models/yolov8n.onnx with ONNX Runtime Web in the
visitor's browser. (Docker/Gradio Spaces on free CPU now need an HF PRO plan, so the
FastAPI service itself is deployed to Vercel; see README.)
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--space", required=True, help="e.g. ANSHUL3977/yolov8-detection-demo")
    a = p.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a Hugging Face token with write access to your repos")
    api = HfApi(token=token)
    api.create_repo(a.space, repo_type="space", space_sdk="static", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copytree(ROOT / "hf-space", stage, dirs_exist_ok=True)
        shutil.copy(ROOT / "models" / "yolov8n.onnx", stage / "yolov8n.onnx")
        shutil.copy(ROOT / "app" / "static" / "bus.jpg", stage / "bus.jpg")
        shutil.copy(ROOT / "app" / "static" / "street.mp4", stage / "street.mp4")
        api.upload_folder(folder_path=stage, repo_id=a.space, repo_type="space", commit_message="Deploy from GitHub")
    print(f"https://huggingface.co/spaces/{a.space}")


if __name__ == "__main__":
    main()
