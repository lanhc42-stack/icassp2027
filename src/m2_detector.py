"""
Table 1, row 2 (retry) -- onset-perturbation leak detector.

First attempt was broken two ways:
  (a) perturbation TRIMMED 1 s, which deletes a second of speech, so transcripts
      differed for a trivial reason rather than because the commitment moved;
  (b) evaluated only on the leakiest tracks, so the negative class was 4-12 clips.

Fixes: PREPEND 1 s of silence (content preserved, onset context changed), and
build a leak-BALANCED set by sweeping SNR from -10 (leaks) to +10 (clean).

Claim under test: if the outcome is decided at onset, perturbing the onset should
flip leaky clips and leave clean clips alone -> disagreement predicts leakage,
with no training and no reference.
"""
import base64
import io
import json
import random
import re
import string
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/opt/tiger/icassp2027/work")
ROOT = Path("/opt/tiger/icassp2027")
DATA, STEMS = ROOT / "data", ROOT / "data" / "musdb16k"
RUNS, OUT = ROOT / "work" / "runs", ROOT / "work" / "m2_detector.json"
WHISPER_DIR = str(ROOT / "models" / "whisper-large-v3-hf")
QWEN_URL = "http://[fdbd:dc55:d:1400::23]:9705/v1/transcribe"

SR, CLIP_S = 16000, 10.0
SNRS = [-10, -5, 0, 5, 10]          # spans leaky -> clean
N_UTTS, N_TRACKS = 3, 10
PREPEND_S = 1.0
SEED = 20260811


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def mix(s, i, snr):
    n = min(len(s), len(i))
    s, i = s[:n].copy(), i[:n].copy()
    i *= (rms(s) / (10 ** (snr / 20.0))) / (rms(i) + 1e-12)
    y = s + i
    pk = np.abs(y).max()
    return (y * (0.99 / pk) if pk > 0.99 else y).astype(np.float32)


def loudest(x, sec=CLIP_S):
    w = int(sec * SR)
    if len(x) <= w:
        return 0
    best, be = 0, -1.0
    for s0 in range(0, len(x) - w, SR // 2):
        e = float(np.sum(x[s0:s0 + w] ** 2))
        if e > be:
            be, best = e, s0
    return best


def norm(s):
    s = (s or "").lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in re.split(r"\s+", s) if w]


solo = {}
for f in ("ctc_ts.json", "qwen3_asr_1.7b_ts.json"):
    for r in json.loads((RUNS / f).read_text()):
        if r["cond"] == "vocals_alone" and r["track"] and \
                len(r["hyp"]) > len(solo.get(r["track"], "")):
            solo[r["track"]] = r["hyp"]
per = {}
for r in json.loads((RUNS / "whisper_large_v3_ts.json").read_text()):
    if r["cond"] == "C2_vocals" and r["snr"] == -10:
        h, sp = norm(r["hyp"]), set(norm(r["ref"]))
        ly = set(norm(solo.get(r["track"], "")))
        per.setdefault(r["track"], []).append(
            len([w for w in h if w in ly and w not in sp]) / max(len(h), 1))
tracks = sorted(per, key=lambda t: -np.mean(per[t]))[:N_TRACKS]

random.seed(SEED)
cands = [f for f in sorted(DATA.glob("LibriSpeech/test-clean/*/*/*.flac"))
         if 8.0 <= sf.info(str(f)).duration <= 14.0]
random.shuffle(cands)
utts = cands[:N_UTTS]
trans = {}
for tf in DATA.glob("LibriSpeech/test-clean/*/*/*.trans.txt"):
    for line in tf.read_text().splitlines():
        uid, text = line.split(" ", 1)
        trans[uid] = text

from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa

wproc = WhisperProcessor.from_pretrained(WHISPER_DIR)
wmodel = WhisperForConditionalGeneration.from_pretrained(
    WHISPER_DIR, torch_dtype=torch.float16).to("cuda").eval()


@torch.no_grad()
def whisper_tx(a):
    f = wproc(a, sampling_rate=SR, return_tensors="pt").input_features.to(
        "cuda", torch.float16)
    return wproc.batch_decode(wmodel.generate(f, max_new_tokens=180),
                              skip_special_tokens=True)[0].strip()


def qwen_tx(a):
    buf = io.BytesIO()
    sf.write(buf, a, SR, format="WAV", subtype="PCM_16")
    payload = json.dumps({"audio": base64.b64encode(buf.getvalue()).decode()}).encode()
    for k in range(3):
        try:
            req = urllib.request.Request(QWEN_URL, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode()).get("text", "") or ""
        except Exception:
            if k == 2:
                return ""
    return ""


BACKENDS = [("whisper", whisper_tx), ("qwen3", qwen_tx)]
pad = np.zeros(int(PREPEND_S * SR), dtype=np.float32)

rows = []
for tr in tracks:
    x, _ = sf.read(str(STEMS / tr / "vocals.wav"), dtype="float32")
    seg = x[loudest(x):][:int(CLIP_S * SR)]
    if len(seg) < int(CLIP_S * SR) * 0.9:
        continue
    ly = set(norm(solo.get(tr, "")))
    for flac in utts:
        spa, _ = sf.read(str(flac), dtype="float32")
        spw = set(norm(trans[flac.stem]))
        for snr in SNRS:
            y = mix(spa, seg, snr)
            yp = np.concatenate([pad, y])
            for bname, fn in BACKENDS:
                t0, t1 = fn(y), fn(yp)
                h = norm(t0)
                lk = len([w for w in h if w in ly and w not in spw]) / max(len(h), 1)
                a, b = set(h), set(norm(t1))
                rows.append({"track": tr, "utt": flac.stem, "snr": snr,
                             "backend": bname, "leak": lk,
                             "disagree": 1.0 - len(a & b) / max(len(a | b), 1)})
        print(f"  {tr[:20]:20s} {flac.stem} done", flush=True)
    OUT.write_text(json.dumps(rows, indent=1))

OUT.write_text(json.dumps(rows, indent=1))
print(f"\nwrote {OUT} ({len(rows)} rows)\n")

print("=== M2 (fixed): prepend-1s perturbation as a training-free leak detector ===")
for bname, _ in BACKENDS:
    rs = [r for r in rows if r["backend"] == bname]
    lk = np.array([r["leak"] for r in rs])
    ds = np.array([r["disagree"] for r in rs])
    pos, neg = ds[lk > 0.3], ds[lk <= 0.3]
    if len(pos) and len(neg):
        auc = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
        print(f"  {bname:9s} AUC={auc:.3f}  disagree leaky={pos.mean():.3f} "
              f"clean={neg.mean():.3f}   n+={len(pos)} n-={len(neg)}")
    print(f"    by SNR: " + "  ".join(
        f"{s:+d}:{np.mean([r['disagree'] for r in rs if r['snr']==s]):.2f}"
        f"/{np.mean([r['leak'] for r in rs if r['snr']==s]):.2f}" for s in SNRS)
        + "   (disagree/leak)")
