"""
Pilot 09 -- the controlled test of H-LM'.

Repeat pilot 08's time-window patching on whisper_ctc_offline (NO decoder).
If the decoder-free model's onset commitment is easier to overturn -- shallower
depth gradient, or a lower plateau -- then "the decode-time prior consolidates
the commitment" stops being an inference from the ladder and becomes a
controlled comparison.

Same tracks/utterances as pilot 08 so the two grids are directly comparable.
CTC encoder: 32 layers, 12 s bucket -> 200 frames (16.7 Hz), so 2 s = 33 frames.
"""
import json
import random
import re
import string
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/opt/tiger/icassp2027/work")
from ctc_torch import (BLANK_ID, SR, ctc_collapse, load_model, log_mel,  # noqa: E402
                       pick_bucket, _tok)

ROOT = Path("/opt/tiger/icassp2027")
DATA, STEMS = ROOT / "data", ROOT / "data" / "musdb16k"
RUNS, OUT = ROOT / "work" / "runs", ROOT / "work" / "patch_ctc.json"

CLIP_S = 10.0
DONOR_SNR, RECV_SNR = 10, -10
N_UTTS, N_TRACKS = 4, 10
SEED = 20260811
ENC_FPS = 200 / 12.0                     # 16.667 Hz for the 12 s bucket
WIN = int(round(2 * ENC_FPS))            # 33 frames = 2 s
N_WIN = 5
REAL_FRAMES = int(round(CLIP_S * ENC_FPS))   # 167
TOTAL_FRAMES = 200
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]


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

# SAME track selection as pilot 08 (Whisper's leakiest) for comparability
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

model = load_model()
blocks = model.layers
print(f"CTC encoder blocks: {len(blocks)}  win={WIN} frames  real={REAL_FRAMES}", flush=True)

cache, plan = {}, None


def mk_hook(idx):
    def hook(mod, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if plan is None:
            cache[idx] = h.detach().clone()
            return output
        L, t0, t1 = plan
        if idx == L and idx in cache:
            h = h.clone()
            h[:, t0:t1, :] = cache[idx][:, t0:t1, :].to(h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return output
    return hook


for i, b in enumerate(blocks):
    b.register_forward_hook(mk_hook(i))


@torch.no_grad()
def run(audio):
    mel = log_mel(audio, pick_bucket(len(audio) / SR)).to("cuda", torch.float16)
    n = mel.shape[-1]
    el = model.conv.calculate_output_length(torch.tensor([n], dtype=torch.int32))
    pos = torch.arange(int(el[0]), dtype=torch.int32, device="cuda").unsqueeze(0)
    lens = torch.tensor([n], dtype=torch.int32, device="cuda")
    yseq = model(mel, pos, lens)[0]
    ids = [i for i in ctc_collapse(yseq[0].tolist()) if i < BLANK_ID]
    return _tok.decode(ids, skip_special_tokens=True).strip()


def record(audio):
    global plan
    plan = None
    cache.clear()
    run(audio)
    return {k: v.clone() for k, v in cache.items()}


def patched(audio, snap, L, t0, t1):
    global plan
    cache.clear()
    cache.update(snap)
    plan = (L, t0, t1)
    txt = run(audio)
    plan = None
    return txt


rows, checks = [], []
for tr in leaky:
    x, _ = sf.read(str(STEMS / tr / "vocals.wav"), dtype="float32")
    seg = x[loudest(x):][:int(CLIP_S * SR)]
    if len(seg) < int(CLIP_S * SR) * 0.9:
        continue
    ly = set(norm(solo.get(tr, "")))
    for flac in utts:
        spa, _ = sf.read(str(flac), dtype="float32")
        spw = set(norm(trans[flac.stem]))

        def leak(t):
            h = norm(t)
            return len([w for w in h if w in ly and w not in spw]) / max(len(h), 1)

        da, ra = mix(spa, seg, DONOR_SNR), mix(spa, seg, RECV_SNR)
        dsnap, rsnap = record(da), record(ra)
        base = leak(run(ra))
        if base < 0.05:
            continue
        checks.append({"base": base,
                       "S1_self": leak(patched(ra, rsnap, 8, 0, TOTAL_FRAMES)),
                       "S2_pad": leak(patched(ra, dsnap, 8, REAL_FRAMES, TOTAL_FRAMES)),
                       "S3_full": leak(patched(ra, dsnap, 8, 0, TOTAL_FRAMES))})
        grid = []
        for L in LAYERS:
            for k in range(N_WIN):
                t0, t1 = k * WIN, min((k + 1) * WIN, REAL_FRAMES)
                grid.append({"layer": L, "win": k,
                             "leak": leak(patched(ra, dsnap, L, t0, t1))})
        rows.append({"track": tr, "utt": flac.stem, "base": base, "grid": grid})
        print(f"  [{len(rows)}] {tr[:24]:24s} base={base:.2f} "
              f"w0L00={grid[0]['leak']:.2f}", flush=True)
    OUT.write_text(json.dumps({"checks": checks, "rows": rows}, indent=1))

OUT.write_text(json.dumps({"checks": checks, "rows": rows}, indent=1))
b = np.mean([c["base"] for c in checks])
print(f"\n=== SANITY (n={len(checks)}) base={b:.3f} "
      f"self={np.mean([c['S1_self'] for c in checks]):.3f} "
      f"pad={np.mean([c['S2_pad'] for c in checks]):.3f} "
      f"full={np.mean([c['S3_full'] for c in checks]):.3f}")
print(f"\n=== CTC: leak after patching (baseline {b:.3f}) ===")
print("        " + "".join(f"  w{k}({k*2}-{k*2+2}s)" for k in range(N_WIN)))
for L in LAYERS:
    line = f"  L{L:02d} "
    for k in range(N_WIN):
        v = [g["leak"] for r in rows for g in r["grid"]
             if g["layer"] == L and g["win"] == k]
        line += f"{np.mean(v):>11.3f}"
    print(line)
