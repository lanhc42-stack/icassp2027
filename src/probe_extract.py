"""
Fig 3 step 1 -- extract Whisper layer activations for the leak probes.

The point: pilot 01 showed the ENCODER retains both sources; Whisper's OUTPUT
contains no lyrics. Probing Whisper's own layers localises where the lyric
content dies -- one model, one training run, so no cross-model confound.

Mixtures are built in memory (deterministic seed) so nothing hits disk.
Hidden states are mean-pooled on-GPU; only pooled vectors reach the CPU.
"""
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path("/opt/tiger/icassp2027")
DATA = ROOT / "data"
STEMS = DATA / "musdb16k"
OUT = ROOT / "work" / "probe_feats.npz"
MODEL = ROOT / "models" / "whisper-large-v3-hf"

SR, CLIP_S = 16000, 10.0
SNRS = [-10, -5]
N_UTTS = 3
SEED = 20260811
CONDS = {"C2_vocals": "vocals.wav", "C1_rhythm": "rhythm.wav"}


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


random.seed(SEED)
cands = [f for f in sorted(DATA.glob("LibriSpeech/test-clean/*/*/*.flac"))
         if 8.0 <= sf.info(str(f)).duration <= 14.0]
random.shuffle(cands)
utts = cands[:N_UTTS]
tracks = sorted(p.name for p in STEMS.iterdir() if p.is_dir())
print(f"{len(utts)} utts x {len(tracks)} tracks x {len(CONDS)} conds x {len(SNRS)} snrs", flush=True)

proc = WhisperProcessor.from_pretrained(str(MODEL))
model = WhisperForConditionalGeneration.from_pretrained(
    str(MODEL), torch_dtype=torch.float16).to("cuda").eval()
n_enc = model.config.encoder_layers
n_dec = model.config.decoder_layers
print(f"whisper: {n_enc} encoder layers, {n_dec} decoder layers", flush=True)

enc_feats, dec_feats, meta = [], [], []


@torch.no_grad()
def features(audio):
    feats = proc(audio, sampling_rate=SR, return_tensors="pt").input_features
    feats = feats.to("cuda", torch.float16)
    enc = model.model.encoder(feats, output_hidden_states=True)
    e = torch.stack([h.mean(dim=1)[0] for h in enc.hidden_states])  # (L+1, D)
    # decode greedily, then re-run the decoder over the produced tokens
    ids = model.generate(feats, max_new_tokens=180)
    dec = model.model.decoder(
        input_ids=ids, encoder_hidden_states=enc.last_hidden_state,
        output_hidden_states=True)
    d = torch.stack([h.mean(dim=1)[0] for h in dec.hidden_states])
    text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return e.float().cpu().numpy(), d.float().cpu().numpy(), text


for ti, tr in enumerate(tracks, 1):
    segs = {}
    for cond, fn in CONDS.items():
        x, _ = sf.read(str(STEMS / tr / fn), dtype="float32")
        st = loudest(x)
        segs[cond] = x[st:st + int(CLIP_S * SR)]
    if min(len(v) for v in segs.values()) < int(CLIP_S * SR) * 0.9:
        continue
    for flac in utts:
        sp, _ = sf.read(str(flac), dtype="float32")
        for cond, seg in segs.items():
            for snr in SNRS:
                e, d, text = features(mix(sp, seg, snr))
                enc_feats.append(e)
                dec_feats.append(d)
                meta.append({"track": tr, "utt": flac.stem, "cond": cond,
                             "snr": snr, "hyp": text})
    if ti % 10 == 0:
        print(f"  [{ti}/{len(tracks)}] {len(meta)} clips", flush=True)

np.savez_compressed(OUT, enc=np.stack(enc_feats), dec=np.stack(dec_feats),
                    meta=json.dumps(meta))
print(f"wrote {OUT}: enc={np.stack(enc_feats).shape} dec={np.stack(dec_feats).shape}")
