"""
Third rung of the ladder: Qwen3-ASR (explicit LLM decoder, song-trained),
served by vLLM on the second pod. Keeps the word timestamps from the bundled
ForcedAligner so leak onset can be located in time.

  MANIFEST_PATH=... TAG_SUFFIX=_ts python3 run_qwen.py
"""
import base64
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "http://[fdbd:dc55:d:1400::23]:9705/v1/transcribe"
ROOT = Path("/opt/tiger/icassp2027")
MANIFEST = Path(os.environ.get("MANIFEST_PATH", ROOT / "work" / "manifest.json"))
SUFFIX = os.environ.get("TAG_SUFFIX", "")
OUT = ROOT / "work" / "runs" / f"qwen3_asr_1.7b{SUFFIX}.json"
WORKERS = int(os.environ.get("WORKERS", "8"))

rows = json.loads(MANIFEST.read_text())
done = [0]
lock = threading.Lock()
t0 = time.time()


def one(r):
    payload = json.dumps({"audio": base64.b64encode(
        Path(r["clip"]).read_bytes()).decode()}).encode()
    hyp, lang, words = "", "", None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                URL, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                d = json.loads(resp.read().decode())
            hyp = d.get("text", "") or ""
            lang = d.get("language", "") or ""
            words = d.get("words")
            break
        except Exception as e:
            if attempt == 2:
                hyp = f"<error: {type(e).__name__}>"
            else:
                time.sleep(2 * (attempt + 1))
    with lock:
        done[0] += 1
        n = done[0]
        if n % 100 == 0 or n == len(rows):
            el = time.time() - t0
            print(f"  [{n}/{len(rows)}] {el:.0f}s ({el/n:.2f}s/clip)", flush=True)
    return {**{k: v for k, v in r.items() if k != "clip"},
            "hyp": hyp, "lang": lang, "words": words}


with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    out = list(ex.map(one, rows))

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
errs = sum(1 for r in out if r["hyp"].startswith("<error"))
print(f"wrote {OUT}  ({len(out)} rows, {errs} errors)")
