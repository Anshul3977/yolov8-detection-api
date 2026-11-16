"""Closed-loop load test: N concurrent clients POST an image to /detect for D seconds.

    uvicorn app.main:app --port 8000 &
    python bench/loadtest.py --url http://127.0.0.1:8000 --concurrency 8 --duration 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


async def worker(client: httpx.AsyncClient, url: str, payload: bytes, deadline: float, lat: list, errors: list):
    while time.perf_counter() < deadline:
        t = time.perf_counter()
        try:
            r = await client.post(url, files={"file": ("bus.jpg", payload, "image/jpeg")})
            if r.status_code == 200:
                lat.append((time.perf_counter() - t) * 1000)
            else:
                errors.append(r.status_code)
        except httpx.HTTPError as exc:
            errors.append(type(exc).__name__)


async def run(a) -> dict:
    payload = Path(a.image).read_bytes()
    lat: list[float] = []
    errors: list = []
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(f"{a.url}/detect", files={"file": ("bus.jpg", payload, "image/jpeg")})  # warm
        start = time.perf_counter()
        deadline = start + a.duration
        await asyncio.gather(*[worker(client, f"{a.url}/detect", payload, deadline, lat, errors) for _ in range(a.concurrency)])
        elapsed = time.perf_counter() - start
    arr = np.asarray(lat)
    return {
        "concurrency": a.concurrency,
        "duration_s": round(elapsed, 1),
        "requests_ok": len(lat),
        "errors": len(errors),
        "rps": round(len(lat) / elapsed, 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "p99_ms": round(float(np.percentile(arr, 99)), 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--duration", type=float, default=30)
    p.add_argument("--image", default=str(ROOT / "tests" / "assets" / "bus.jpg"))
    p.add_argument("--out")
    a = p.parse_args()
    result = asyncio.run(run(a))
    print(json.dumps(result, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
