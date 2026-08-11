"""
Pilot 10 -- close the frame-count confound in the pilot 08 vs 09 comparison.

Pilot 08/09 aligned windows by TIME (2 s each). But the encoders run at
different rates, so a 2 s window replaced 100 vectors in Whisper and only 33 in
CTC. Whisper's much larger onset advantage could therefore be an artefact of
replacing 3x more vectors.

Here: run Whisper with 33-FRAME windows -- the same vector count CTC got --
placed at the start of each 2 s segment so relative position is preserved.

  Whisper encoder 50 Hz  -> 33 frames = 0.66 s, at frames 0/100/200/300/400
  CTC encoder     16.7 Hz-> 33 frames = 2.00 s  (pilot 09)

If Whisper's onset advantage survives at matched vector count, the mechanism
claim holds and the confound is closed.
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
RUNS, OUT = ROOT / "work" / "runs", ROOT / "work" / "patch_framematched.json"

SR, CLIP_S = 16000, 10.0
DONOR_SNR, RECV_SNR = 10, -10
N_UTTS, N_TRACKS = 4, 10
SEED = 20260811
WIN = 33                                  # matched to CTC's 2 s = 33 frames
STARTS = [0, 100, 200, 300, 400]          # start of each 2 s segment at 50 Hz
REAL_FRAMES = 500                         # 10 s at 50 Hz
TOTAL = 1500
LAYERS = [0, 4, 8, 16, 24, 31]


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
print(f"WIN={WIN} frames = {WIN/50:.2f}s of Whisper audio", flush=True)

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

        dfe, rfe = feats_of(mix(spa, seg, DONOR_SNR)), feats_of(mix(spa, seg, RECV_SNR))
        dsnap, rsnap = record(dfe), record(rfe)
        base = leak(decode(rfe))
        if base < 0.05:
            continue
        checks.append({"base": base,
                       "S1_self": leak(patched(rfe, rsnap, 8, 0, TOTAL)),
                       "S2_pad": leak(patched(rfe, dsnap, 8, REAL_FRAMES, TOTAL)),
                       "S3_full": leak(patched(rfe, dsnap, 8, 0, TOTAL))})
        grid = []
        for L in LAYERS:
            for k, s0 in enumerate(STARTS):
                grid.append({"layer": L, "win": k,
                             "leak": leak(patched(rfe, dsnap, L, s0, s0 + WIN))})
        rows.append({"track": tr, "utt": flac.stem, "base": base, "grid": grid})
        print(f"  [{len(rows)}] {tr[:22]:22s} base={base:.2f} w0L00={grid[0]['leak']:.2f}",
              flush=True)
    OUT.write_text(json.dumps({"checks": checks, "rows": rows}, indent=1))

OUT.write_text(json.dumps({"checks": checks, "rows": rows}, indent=1))
b = np.mean([c["base"] for c in checks])
print(f"\n=== SANITY (n={len(checks)}) base={b:.3f} "
      f"self={np.mean([c['S1_self'] for c in checks]):.3f} "
      f"pad={np.mean([c['S2_pad'] for c in checks]):.3f} "
      f"full={np.mean([c['S3_full'] for c in checks]):.3f}")
print(f"\n=== Whisper, {WIN}-frame windows (baseline {b:.3f}) ===")
print("        " + "".join(f"  w{k}@{s/50:.1f}s" for k, s in enumerate(STARTS)))
means = {}
for L in LAYERS:
    line = f"  L{L:02d} "
    for k in range(len(STARTS)):
        v = [g["leak"] for r in rows for g in r["grid"]
             if g["layer"] == L and g["win"] == k]
        means[(L, k)] = np.mean(v)
        line += f"{np.mean(v):>10.3f}"
    print(line)

L0 = LAYERS[0]
onset = (b - means[(L0, 0)]) / b * 100
other = np.mean([(b - means[(L0, k)]) / b * 100 for k in range(1, len(STARTS))])
print(f"\n  at L{L0:02d}: onset reduction {onset:.1f}%  others {other:.1f}%  "
      f"advantage {onset-other:+.1f} pts")
print("  (pilot 09 CTC, 33 frames: onset 45.6%  others 36.7%  advantage +8.9 pts)")
