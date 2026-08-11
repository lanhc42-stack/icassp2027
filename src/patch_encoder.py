"""
Fig 3/4 -- causal localisation by activation patching (encoder).

Matched pair: SAME speech, SAME track, only the mix ratio differs.
  donor     SNR +10 -> Whisper emits speech, no leak
  receiver  SNR -10 -> Whisper leaks lyrics

For each encoder layer L we run the receiver but overwrite layer L's output with
the donor's, then decode. If the leak disappears when patching at L, the leak is
caused by information carried at L.

Encoder activations are fixed-length (1500 frames) regardless of decoding, so
there is no alignment problem -- unlike decoder patching.
"""
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path("/opt/tiger/icassp2027")
DATA, STEMS = ROOT / "data", ROOT / "data" / "musdb16k"
MODEL = ROOT / "models" / "whisper-large-v3-hf"
RUNS = ROOT / "work" / "runs"
OUT = ROOT / "work" / "patch_encoder.json"

SR, CLIP_S = 16000, 10.0
DONOR_SNR, RECV_SNR = 10, -10
N_UTTS, N_TRACKS = 3, 8
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
    import re, string
    s = (s or "").lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in re.split(r"\s+", s) if w]


# lyric reference per track (best available transcript of the isolated vocals)
solo = {}
for f in ("ctc_ts.json", "qwen3_asr_1.7b_ts.json"):
    for r in json.loads((RUNS / f).read_text()):
        if r["cond"] == "vocals_alone" and r["track"]:
            if len(r["hyp"]) > len(solo.get(r["track"], "")):
                solo[r["track"]] = r["hyp"]

# tracks where Whisper actually leaks -- patching is only meaningful there
wl = json.loads((RUNS / "whisper_large_v3_ts.json").read_text())
per = {}
for r in wl:
    if r["cond"] == "C2_vocals" and r["snr"] == -10:
        h, sp = norm(r["hyp"]), set(norm(r["ref"]))
        ly = set(norm(solo.get(r["track"], "")))
        v = len([w for w in h if w in ly and w not in sp]) / max(len(h), 1)
        per.setdefault(r["track"], []).append(v)
leaky = sorted(per, key=lambda t: -np.mean(per[t]))[:N_TRACKS]
print("patching tracks (Whisper leak @-10):",
      [f"{t[:24]}={np.mean(per[t]):.2f}" for t in leaky], flush=True)

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
NL = len(layers)
print(f"encoder layers: {NL}", flush=True)

_donor_cache, _patch_at = {}, None


def mk_hook(idx):
    def hook(mod, args, output):
        if _patch_at == idx and idx in _donor_cache:
            h = output[0] if isinstance(output, tuple) else output
            new = _donor_cache[idx].to(h.dtype)
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        if _patch_at is None:                       # recording pass
            h = output[0] if isinstance(output, tuple) else output
            _donor_cache[idx] = h.detach().clone()
        return output
    return hook


for i, l in enumerate(layers):
    l.register_forward_hook(mk_hook(i))


@torch.no_grad()
def feats_of(audio):
    f = proc(audio, sampling_rate=SR, return_tensors="pt").input_features
    return f.to("cuda", torch.float16)


@torch.no_grad()
def decode(feats):
    ids = model.generate(feats, max_new_tokens=180)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


rows = []
for ti, tr in enumerate(leaky, 1):
    x, _ = sf.read(str(STEMS / tr / "vocals.wav"), dtype="float32")
    seg = x[loudest(x):][:int(CLIP_S * SR)]
    if len(seg) < int(CLIP_S * SR) * 0.9:
        continue
    ly = set(norm(solo.get(tr, "")))
    for flac in utts:
        sp_audio, _ = sf.read(str(flac), dtype="float32")
        ref = trans[flac.stem]
        spw = set(norm(ref))

        def leak_of(text):
            h = norm(text)
            return len([w for w in h if w in ly and w not in spw]) / max(len(h), 1)

        donor_f = feats_of(mix(sp_audio, seg, DONOR_SNR))
        recv_f = feats_of(mix(sp_audio, seg, RECV_SNR))

        globals()["_patch_at"] = None
        _donor_cache.clear()
        donor_txt = decode(donor_f)                 # records donor activations
        base_txt = decode(recv_f)                   # NOTE: overwrites cache
        # re-record donor cleanly, then patch
        globals()["_patch_at"] = None
        _donor_cache.clear()
        decode(donor_f)
        donor_snapshot = {k: v.clone() for k, v in _donor_cache.items()}

        base_leak = leak_of(base_txt)
        if base_leak < 0.05:                        # nothing to remove
            continue
        per_layer = []
        for L in range(NL):
            _donor_cache.clear()
            _donor_cache.update(donor_snapshot)
            globals()["_patch_at"] = L
            txt = decode(recv_f)
            per_layer.append({"layer": L, "leak": leak_of(txt), "text": txt[:110]})
        globals()["_patch_at"] = None

        rows.append({"track": tr, "utt": flac.stem, "base_leak": base_leak,
                     "donor_leak": leak_of(donor_txt), "base_text": base_txt[:140],
                     "patched": per_layer})
        print(f"  {tr[:26]:26s} {flac.stem} base={base_leak:.2f} "
              f"min_patched={min(p['leak'] for p in per_layer):.2f}", flush=True)
    if ti % 2 == 0:
        OUT.write_text(json.dumps(rows, indent=1))

OUT.write_text(json.dumps(rows, indent=1))
print(f"\nwrote {OUT} ({len(rows)} pairs)")

if rows:
    print("\nlayer  mean_leak_after_patch   (baseline "
          f"{np.mean([r['base_leak'] for r in rows]):.3f})")
    for L in range(NL):
        v = [r["patched"][L]["leak"] for r in rows]
        print(f"  L{L:02d}   {np.mean(v):.3f}")
