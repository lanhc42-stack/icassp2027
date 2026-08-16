#!/usr/bin/env python3
"""Small local HTTP adapter for Qwen3-ASR used by source_capture."""

from __future__ import annotations

import base64
import io
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel


MODEL_PATH = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
ALIGNER_PATH = os.environ.get(
    "QWEN_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B"
)
DTYPE = getattr(torch, os.environ.get("QWEN_DTYPE", "bfloat16"))
HOST = os.environ.get("QWEN_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("QWEN_SERVER_PORT", "9705"))
BATCH_SIZE = int(os.environ.get("QWEN_BATCH_SIZE", "4"))
BATCH_WAIT_MS = float(os.environ.get("QWEN_BATCH_WAIT_MS", "20"))


model = Qwen3ASRModel.from_pretrained(
    MODEL_PATH,
    dtype=DTYPE,
    device_map="cuda:0",
    max_inference_batch_size=BATCH_SIZE,
    max_new_tokens=256,
    forced_aligner=ALIGNER_PATH,
    forced_aligner_kwargs={"dtype": DTYPE, "device_map": "cuda:0"},
)


@dataclass
class BatchItem:
    audio: tuple[object, int]
    done: threading.Event = field(default_factory=threading.Event)
    response: dict[str, object] | None = None
    error: BaseException | None = None


request_queue: queue.Queue[BatchItem] = queue.Queue()


def _value(item: object, name: str) -> object | None:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _format_result(result: object) -> dict[str, object]:
    words = []
    for item in getattr(result, "time_stamps", None) or []:
        word = _value(item, "text")
        start = _value(item, "start_time")
        end = _value(item, "end_time")
        if word is None or start is None or end is None:
            continue
        words.append(
            {"word": str(word), "start": float(start), "end": float(end)}
        )
    return {
        "text": str(getattr(result, "text", "")),
        "language": str(getattr(result, "language", "")),
        "words": words or None,
    }


def _batch_worker() -> None:
    while True:
        first = request_queue.get()
        items = [first]
        deadline = time.monotonic() + BATCH_WAIT_MS / 1000.0
        while len(items) < BATCH_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                items.append(request_queue.get(timeout=remaining))
            except queue.Empty:
                break
        try:
            results = model.transcribe(
                audio=[item.audio for item in items],
                language=None,
                return_time_stamps=True,
            )
            if len(results) != len(items):
                raise RuntimeError(
                    f"Qwen returned {len(results)} results for {len(items)} inputs"
                )
            for item, result in zip(items, results):
                item.response = _format_result(result)
        except BaseException as exc:
            for item in items:
                item.error = exc
        finally:
            for item in items:
                item.done.set()
                request_queue.task_done()


def _transcribe(audio_bytes: bytes) -> dict[str, object]:
    audio, sample_rate = sf.read(
        io.BytesIO(audio_bytes), dtype="float32", always_2d=False
    )
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)
    item = BatchItem(audio=(audio, int(sample_rate)))
    request_queue.put(item)
    item.done.wait()
    if item.error is not None:
        raise item.error
    assert item.response is not None
    return item.response


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            audio_bytes = base64.b64decode(payload["audio"], validate=True)
            response = _transcribe(audio_bytes)
            body = json.dumps(response, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
        except Exception as exc:
            body = json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
            ).encode("utf-8")
            self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(format % args, flush=True)


if __name__ == "__main__":
    print(f"loading Qwen3-ASR from {MODEL_PATH}", flush=True)
    threading.Thread(target=_batch_worker, name="qwen-batch-worker", daemon=True).start()
    print(
        f"serving on http://{HOST}:{PORT} "
        f"with batch_size={BATCH_SIZE}, batch_wait_ms={BATCH_WAIT_MS:g}",
        flush=True,
    )
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
