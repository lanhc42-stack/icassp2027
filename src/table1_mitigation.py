"""
Table 1 -- mitigations derived from the mechanism (pilots 08-10).

M1  ONSET DUCKING.  The commitment forms in the first ~2 s, so attenuating the
    interferer only during those 2 s should fix the WHOLE clip. Compare against
    ducking the whole clip (upper bound) and ducking the LAST 2 s (position
    control). If onset ducking recovers most of the benefit at 20% of the
    processing, that is a deployable result.

M2  SHIFT-DISAGREEMENT DETECTOR.  If the outcome is decided at onset, giving the
    model a different onset should change the outcome precisely on clips that
    leak, and leave clean clips alone. Transcribe as-is and with 1 s trimmed;
    disagreement between the two flags leakage. Training-free, model-agnostic.

Models: Whisper large-v3 and whisper_ctc_offline locally; Qwen3-ASR over HTTP.
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
RUNS, OUT = ROOT / "work" / "runs", ROOT / "work" / "table1.json"
WHISPER_DIR = str(ROOT / "models" / "whisper-large-v3-hf")
QWEN_URL = "http://[fdbd:dc55:d:1400::23]:9705/v1/transcribe"

SR, CLIP_S = 16000, 10.0
BASE_SNR, DUCK_SNR = -10, 10
ONSET_S = 2.0
N_UTTS, N_TRACKS = 4, 12
SEED = 20260811


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def scaled_interferer(s, i, snr):
    n = min(len(s), len(i))
    i = i[:n].copy()
    i *= (rms(s[:n]) / (10 ** (snr / 20.0))) / (rms(i) + 1e-12)
    return i


def compose(s, i, gain_env):
    n = min(len(s), len(i), len(gain_env))
    y = s[:n] + i[:n] * gain_env[:n]
    pk = np.abs(y).max()
    return (y * (0.99 / pk) if pk > 0.99 else y).astype(np.float32)


def env_const(n, g):
    return np.full(n, g, dtype=np.float32)


def env_duck(n, start_s, dur_s, g):
    e = np.ones(n, dtype=np.float32)
    a, b = int(start_s * SR), int((start_s + dur_s) * SR)
    e[a:b] = g
    return e


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


# ---------------------------------------------------------------- references
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
leaky = sorted(per, key=lambda t: -np.mean(per[t]))[:N_TRACKS]

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

# ------------------------------------------------------------------ backends
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


import ctc_torch  # noqa: E402

cmodel = ctc_torch.load_model()


def ctc_tx(a):
    return ctc_torch.transcribe(cmodel, a)


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


BACKENDS = [("whisper", whisper_tx), ("ctc", ctc_tx), ("qwen3", qwen_tx)]

# ---------------------------------------------------------------------- main
rows = []
for tr in leaky:
    x, _ = sf.read(str(STEMS / tr / "vocals.wav"), dtype="float32")
    seg = x[loudest(x):][:int(CLIP_S * SR)]
    if len(seg) < int(CLIP_S * SR) * 0.9:
        continue
    ly = set(norm(solo.get(tr, "")))
    for flac in utts:
        spa, _ = sf.read(str(flac), dtype="float32")
        ref = trans[flac.stem]
        spw = set(norm(ref))
        n = min(len(spa), len(seg))
        s, i = spa[:n], scaled_interferer(spa, seg, BASE_SNR)[:n]
        duck_g = 10 ** ((BASE_SNR - DUCK_SNR) / 20.0)     # extra attenuation

        conds = {
            "baseline": compose(s, i, env_const(n, 1.0)),
            "duck_onset": compose(s, i, env_duck(n, 0.0, ONSET_S, duck_g)),
            "duck_last": compose(s, i, env_duck(n, CLIP_S - ONSET_S, ONSET_S, duck_g)),
            "duck_all": compose(s, i, env_const(n, duck_g)),
        }

        def leak(t):
            h = norm(t)
            return len([w for w in h if w in ly and w not in spw]) / max(len(h), 1)

        def recall(t):
            h = set(norm(t))
            return len([w for w in spw if w in h]) / max(len(spw), 1)

        for bname, fn in BACKENDS:
            out = {}
            for cname, audio in conds.items():
                t = fn(audio)
                out[cname] = {"leak": leak(t), "recall": recall(t), "text": t[:100]}
            # M2 detector: same clip, 1 s of onset removed
            shifted = fn(conds["baseline"][int(1.0 * SR):])
            b = set(norm(out["baseline"]["text"]))
            sh = set(norm(shifted))
            jac = len(b & sh) / max(len(b | sh), 1)
            out["shift_disagreement"] = 1.0 - jac
            rows.append({"track": tr, "utt": flac.stem, "backend": bname, **out})
        print(f"  {tr[:22]:22s} {flac.stem} done", flush=True)
    OUT.write_text(json.dumps(rows, indent=1))

OUT.write_text(json.dumps(rows, indent=1))
print(f"\nwrote {OUT} ({len(rows)} rows)\n")

print("=== Table 1: onset ducking (leak / speech recall) ===")
print(f"{'model':10s}{'baseline':>18s}{'duck onset 2s':>18s}{'duck last 2s':>18s}{'duck all':>18s}")
for bname, _ in BACKENDS:
    rs = [r for r in rows if r["backend"] == bname]
    if not rs:
        continue
    line = f"{bname:10s}"
    for c in ("baseline", "duck_onset", "duck_last", "duck_all"):
        lk = np.mean([r[c]["leak"] for r in rs])
        rc = np.mean([r[c]["recall"] for r in rs])
        line += f"{lk:8.3f}/{rc:<9.3f}"
    print(line)

print("\n=== fraction of the full-ducking benefit recovered by onset-only ===")
for bname, _ in BACKENDS:
    rs = [r for r in rows if r["backend"] == bname]
    if not rs:
        continue
    b = np.mean([r["baseline"]["leak"] for r in rs])
    o = np.mean([r["duck_onset"]["leak"] for r in rs])
    l_ = np.mean([r["duck_last"]["leak"] for r in rs])
    a = np.mean([r["duck_all"]["leak"] for r in rs])
    denom = (b - a) if (b - a) > 1e-6 else float("nan")
    print(f"  {bname:10s} onset {100*(b-o)/denom:5.1f}%   last-2s control "
          f"{100*(b-l_)/denom:5.1f}%   (processing cost 20% of clip)")

print("\n=== M2: shift-disagreement as a leak detector ===")
for bname, _ in BACKENDS:
    rs = [r for r in rows if r["backend"] == bname]
    if not rs:
        continue
    lk = np.array([r["baseline"]["leak"] for r in rs])
    ds = np.array([r["shift_disagreement"] for r in rs])
    pos, neg = ds[lk > 0.3], ds[lk <= 0.3]
    if len(pos) and len(neg):
        auc = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
        print(f"  {bname:10s} AUC={auc:.3f}  disagreement leaky={pos.mean():.3f} "
              f"clean={neg.mean():.3f}  (n+={len(pos)} n-={len(neg)})")
