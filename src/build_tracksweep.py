"""
Track-variance sweep: all 50 MUSDB test tracks, C2_vocals only, 3 SNRs.
Pilot 02 showed track identity dominates SNR; 6 tracks is too few to claim it.
"""
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("/opt/tiger/icassp2027")
DATA = ROOT / "data"
STEMS = DATA / "musdb16k"
CLIPS = ROOT / "work" / "clips_ts"
MANIFEST = ROOT / "work" / "manifest_ts.json"
SR, CLIP_S = 16000, 10.0
SNRS = [-10, -5, 0]
N_UTTS = 5
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


random.seed(SEED)
CLIPS.mkdir(parents=True, exist_ok=True)
trans = {}
for tf in DATA.glob("LibriSpeech/test-clean/*/*/*.trans.txt"):
    for line in tf.read_text().splitlines():
        uid, text = line.split(" ", 1)
        trans[uid] = text

cands = [f for f in sorted(DATA.glob("LibriSpeech/test-clean/*/*/*.flac"))
         if 8.0 <= sf.info(str(f)).duration <= 14.0]
random.shuffle(cands)
utts = cands[:N_UTTS]
tracks = sorted(p.name for p in STEMS.iterdir() if p.is_dir())
print(f"{len(utts)} utts x {len(tracks)} tracks x {len(SNRS)} snrs", flush=True)

man = []
for tr in tracks:
    x, _ = sf.read(str(STEMS / tr / "vocals.wav"), dtype="float32")
    seg = x[loudest(x):][:int(CLIP_S * SR)]
    if len(seg) < int(CLIP_S * SR) * 0.9:
        print(f"  skip (too short): {tr}", flush=True)
        continue
    safe = tr.replace(" ", "_").replace("/", "_")[:40]
    p = CLIPS / f"CTRL__{safe}__vocals_alone.wav"
    sf.write(p, seg, SR, subtype="PCM_16")
    man.append({"clip": str(p), "utt": None, "ref": None, "track": tr,
                "cond": "vocals_alone", "snr": None})
    for flac in utts:
        uid = flac.stem
        sp, _ = sf.read(str(flac), dtype="float32")
        for snr in SNRS:
            p = CLIPS / f"{uid}__{safe}__C2_vocals__{snr:+d}.wav"
            sf.write(p, mix(sp, seg, snr), SR, subtype="PCM_16")
            man.append({"clip": str(p), "utt": uid, "ref": trans[uid],
                        "track": tr, "cond": "C2_vocals", "snr": snr})

MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1))
print(f"wrote {len(man)} clips -> {MANIFEST}")
