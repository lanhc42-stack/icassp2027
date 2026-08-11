"""
Fig 3 -- 200 ms resolution inside the onset window.

Pilot 06 put the commitment in the first 2 s and encoder layers <= 12, at 2 s
resolution. Here: 200 ms cells (10 encoder frames at 50 Hz) across 0-2 s, plus
SIZE-MATCHED control cells at 4.0-4.2 s and 8.0-8.2 s so we can tell whether a
200 ms patch does anything at all away from the onset.

Same sanity checks as pilot 06 (self / padding / full).
"""
import json
import random
import re
import string
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path("/opt/tiger/icassp2027")
DATA, STEMS = ROOT / "data", ROOT / "data" / "musdb16k"
MODEL = ROOT / "models" / "whisper-large-v3-hf"
RUNS, OUT = ROOT / "work" / "runs", ROOT / "work" / "patch_fine.json"

SR, CLIP_S = 16000, 10.0
DONOR_SNR, RECV_SNR = 10, -10
N_UTTS, N_TRACKS = 4, 10
SEED = 20260811
FPS = 50                                  # encoder frames per second
CELL = 10                                 # 10 frames = 200 ms
LAYERS = [0, 4, 8, 12, 16, 24]
ONSET_CELLS = [(k * CELL, (k + 1) * CELL) for k in range(10)]       # 0..2 s
CONTROL_CELLS = [(4 * FPS, 4 * FPS + CELL), (8 * FPS, 8 * FPS + CELL)]


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

proc = WhisperProcessor.from_pretrained(str(MODEL))
model = WhisperForConditionalGeneration.from_pretrained(
    str(MODEL), torch_dtype=torch.float16).to("cuda").eval()
layers = model.model.encoder.layers

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


for i, l in enumerate(layers):
    l.register_forward_hook(mk_hook(i))


@torch.no_grad()
def feats_of(a):
    return proc(a, sampling_rate=SR, return_tensors="pt").input_features.to(
        "cuda", torch.float16)


@torch.no_grad()
def decode(f):
    return proc.batch_decode(model.generate(f, max_new_tokens=180),
                             skip_special_tokens=True)[0].strip()


def record(feats):
    global plan
    plan = None
    cache.clear()
    decode(feats)
    return {k: v.clone() for k, v in cache.items()}


def patched(feats, snap, L, t0, t1):
    global plan
    cache.clear()
    cache.update(snap)
    plan = (L, t0, t1)
    txt = decode(feats)
    plan = None
    return txt


rows, checks = [], []
for ti, tr in enumerate(leaky, 1):
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

        dfe = feats_of(mix(spa, seg, DONOR_SNR))
        rfe = feats_of(mix(spa, seg, RECV_SNR))
        dsnap, rsnap = record(dfe), record(rfe)
        base = leak(decode(rfe))
        if base < 0.05:
            continue
        checks.append({"base": base,
                       "S1_self": leak(patched(rfe, rsnap, 8, 0, 1500)),
                       "S2_pad": leak(patched(rfe, dsnap, 8, int(CLIP_S * FPS), 1500)),
                       "S3_full": leak(patched(rfe, dsnap, 8, 0, 1500))})

        grid = []
        for L in LAYERS:
            for (t0, t1) in ONSET_CELLS:
                grid.append({"layer": L, "kind": "onset", "t0": t0,
                             "leak": leak(patched(rfe, dsnap, L, t0, t1))})
            for (t0, t1) in CONTROL_CELLS:
                grid.append({"layer": L, "kind": "control", "t0": t0,
                             "leak": leak(patched(rfe, dsnap, L, t0, t1))})
        rows.append({"track": tr, "utt": flac.stem, "base": base, "grid": grid})
        print(f"  [{len(rows)}] {tr[:24]:24s} base={base:.2f} "
              f"min_onset={min(g['leak'] for g in grid if g['kind']=='onset'):.2f}",
              flush=True)
    if ti % 2 == 0:
        OUT.write_text(json.dumps({"checks": checks, "rows": rows}, indent=1))

OUT.write_text(json.dumps({"checks": checks, "rows": rows}, indent=1))
print(f"\nwrote {OUT} ({len(rows)} pairs)")

b = np.mean([c["base"] for c in checks])
print(f"\n=== SANITY (n={len(checks)}) base={b:.3f} "
      f"self={np.mean([c['S1_self'] for c in checks]):.3f} "
      f"pad={np.mean([c['S2_pad'] for c in checks]):.3f} "
      f"full={np.mean([c['S3_full'] for c in checks]):.3f}")

print(f"\n=== residual leak, 200 ms cells (baseline {b:.3f}) ===")
hdr = "".join(f"{t0*1000//FPS:>7d}" for t0, _ in ONSET_CELLS)
print(f"  layer{hdr}   |  ctrl4s  ctrl8s")
for L in LAYERS:
    line = f"  L{L:02d}  "
    for (t0, _) in ONSET_CELLS:
        v = [g["leak"] for r in rows for g in r["grid"]
             if g["layer"] == L and g["kind"] == "onset" and g["t0"] == t0]
        line += f"{np.mean(v):>7.3f}" if v else "      -"
    cs = []
    for (t0, _) in CONTROL_CELLS:
        v = [g["leak"] for r in rows for g in r["grid"]
             if g["layer"] == L and g["kind"] == "control" and g["t0"] == t0]
        cs.append(np.mean(v) if v else float("nan"))
    print(line + f"   | {cs[0]:7.3f} {cs[1]:7.3f}")
