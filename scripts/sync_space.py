"""Push this repo to a Docker Hugging Face Space.

Does nothing when HF_TOKEN or HF_SPACE is unset, so the GitHub Action is a
no-op until Anshul adds them. The Space README gets the Docker SDK header
even though the GitHub README stays free of that YAML table.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HEADER = """---
title: YOLOv8 Detection API
emoji: "🎯"
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
short_description: YOLOv8n detection on ONNX Runtime, CPU
---

"""

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    space = os.environ.get("HF_SPACE", "").strip()
    if not token or not space:
        print("HF_TOKEN or HF_SPACE is unset. Skipping Space sync.")
        return 0
    if space.count("/") != 1:
        print("HF_SPACE must look like username/space-name.", file=sys.stderr)
        return 1

    readme = ROOT / "README.md"
    original = readme.read_text()
    if "sdk: docker" not in original[:500]:
        readme.write_text(HEADER + original)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.upload_folder(
            repo_id=space,
            repo_type="space",
            folder_path=str(ROOT),
            commit_message="Sync from github.com/Anshul3977/yolov8-detection-api",
            ignore_patterns=[
                ".git/**",
                ".github/**",
                ".venv/**",
                "**/__pycache__/**",
                ".pytest_cache/**",
                ".ruff_cache/**",
                "*.pt",
            ],
        )
    finally:
        readme.write_text(original)
    print(f"Synced to https://huggingface.co/spaces/{space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
