"""Push the serving files to a Hugging Face Docker Space.

    pip install huggingface_hub
    HF_TOKEN=hf_xxx python scripts/deploy_hf_space.py --space <hf-username>/yolov8-detection-api

Needs a token with write access. Creates the Space if it doesn't exist.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
FILES = ["Dockerfile", "requirements.txt", ".dockerignore"]
DIRS = ["app", "models"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--space", required=True, help="e.g. anshul3977/yolov8-detection-api")
    a = p.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a Hugging Face token with write scope")

    api = HfApi(token=token)
    api.create_repo(a.space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for f in FILES:
            shutil.copy(ROOT / f, stage / f)
        for d in DIRS:
            shutil.copytree(ROOT / d, stage / d, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "deploy" / "space_README.md", stage / "README.md")
        api.upload_folder(folder_path=stage, repo_id=a.space, repo_type="space", commit_message="Deploy from GitHub")
    print(f"https://huggingface.co/spaces/{a.space}")


if __name__ == "__main__":
    main()
