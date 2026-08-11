"""
Time-window activation patching, with discriminating-power checks up front.

Whole-layer patching was uninformative (pilot 05 attempt C): replacing a whole
layer output makes everything downstream a deterministic function of the donor.
Patching a TIME WINDOW leaves the receiver's other frames in place, so the
residual stream still carries receiver information and localisation is possible
in depth AND time.

Whisper: 30 s padded input -> 3000 mel frames -> 1500 encoder frames (50 Hz).
A 10 s clip occupies frames 0..500; 500..1500 is padding.

SANITY CHECKS (run first, must all pass or the sweep is meaningless):
  S1 self-patch      receiver into itself, all frames -> leak must STAY ~baseline
  S2 padding-patch   donor frames 500..1500 only      -> leak must STAY ~baseline
  S3 full-patch      donor all frames                 -> leak must DROP to ~0
S1 catches a broken harness; S2 is the real test that the manipulation is not
just "run the donor"; S3 confirms the donor genuinely suppresses.
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
RUNS, OUT = ROOT / "work" / "runs", ROOT / "work" / "patch_window_n39.json"

SR, CLIP_S = 16000, 10.0
DONOR_SNR, RECV_SNR = 10, -10
N_UTTS, N_TRACKS = 4, 10
SEED = 20260811
REAL_FRAMES = int(CLIP_S * 50)          # 10 s at 50 Hz = 500
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
N_WIN = 5                                # 5 windows x 100 frames = 2 s each


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

cache, plan = {}, None      # plan = (layer, t0, t1) or None


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

        dfe = feats_of(mix(spa, seg, DONOR_SNR))
        rfe = feats_of(mix(spa, seg, RECV_SNR))
        dsnap = record(dfe)
        rsnap = record(rfe)
        base = leak(decode(rfe))
        if base < 0.05:
            continue

        s1 = leak(patched(rfe, rsnap, 16, 0, 1500))          # self-patch
        s2 = leak(patched(rfe, dsnap, 16, REAL_FRAMES, 1500))  # padding only
        s3 = leak(patched(rfe, dsnap, 16, 0, 1500))          # full donor
        checks.append({"track": tr, "utt": flac.stem, "base": base,
                       "S1_self": s1, "S2_pad": s2, "S3_full": s3})
        print(f"  SANITY {tr[:22]:22s} base={base:.2f} "
              f"S1_self={s1:.2f} S2_pad={s2:.2f} S3_full={s3:.2f}", flush=True)

        grid = []
        w = REAL_FRAMES // N_WIN
        for L in LAYERS:
            for k in range(N_WIN):
                t0, t1 = k * w, (k + 1) * w
                grid.append({"layer": L, "win": k, "t0": t0, "t1": t1,
                             "leak": leak(patched(rfe, dsnap, L, t0, t1))})
        rows.append({"track": tr, "utt": flac.stem, "base": base, "grid": grid})
        print(f"    swept {len(grid)} cells, min={min(g['leak'] for g in grid):.2f}",
              flush=True)

OUT.write_text(json.dumps({"checks": checks, "rows": rows}, indent=1))
print(f"\nwrote {OUT}")

if checks:
    print("\n=== SANITY (n=%d) ===" % len(checks))
    for k in ("base", "S1_self", "S2_pad", "S3_full"):
        print(f"  {k:9s} {np.mean([c[k] for c in checks]):.3f}")
    ok = (np.mean([c["S1_self"] for c in checks]) > 0.7 * np.mean([c["base"] for c in checks])
          and np.mean([c["S2_pad"] for c in checks]) > 0.7 * np.mean([c["base"] for c in checks])
          and np.mean([c["S3_full"] for c in checks]) < 0.2 * np.mean([c["base"] for c in checks]))
    print(f"  => discriminating power: {'PASS' if ok else 'FAIL'}")

if rows:
    print("\n=== leak after patching (rows=layer, cols=2s window) ===")
    print("       " + "".join(f"  w{k}({k*2}-{k*2+2}s)" for k in range(N_WIN)))
    for L in LAYERS:
        line = f"  L{L:02d} "
        for k in range(N_WIN):
            v = [g["leak"] for r in rows for g in r["grid"]
                 if g["layer"] == L and g["win"] == k]
            line += f"{np.mean(v):>11.3f}"
        print(line)
    print(f"\n  baseline {np.mean([r['base'] for r in rows]):.3f}")
