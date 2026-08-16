#!/usr/bin/env python3
"""Benchmark a Qwen HTTP endpoint against archived single-item predictions."""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def request(url: str, row: dict[str, object]) -> dict[str, object]:
    payload = json.dumps(
        {
            "audio": base64.b64encode(
                Path(str(row["audio_path"])).read_bytes()
            ).decode("ascii")
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.runs.read_text().splitlines() if line]
    selected = rows[-args.samples :]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda row: request(args.url, row), selected))
    elapsed = time.perf_counter() - started

    exact = sum(
        str(result.get("text", "")).strip() == str(row.get("hyp", "")).strip()
        for row, result in zip(selected, results)
    )
    print(f"samples={len(selected)} workers={args.workers} elapsed_s={elapsed:.3f}")
    print(f"throughput_samples_s={len(selected) / elapsed:.4f}")
    print(f"exact_text_matches={exact}/{len(selected)}")
    for row, result in zip(selected, results):
        expected = str(row.get("hyp", "")).strip()
        actual = str(result.get("text", "")).strip()
        if actual != expected:
            print(json.dumps({"sample_id": row.get("sample_id"), "expected": expected, "actual": actual}, ensure_ascii=False))


if __name__ == "__main__":
    main()
