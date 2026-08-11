"""
Materialise one fixed mixture set so every model on the prior-strength ladder
sees byte-identical audio. Uses the properly-resampled 16k stems (musdb16k),
not the pilot's linear-interp path.

Writes clips/*.wav + manifest.json
"""
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("/opt/tiger/icassp2027")
DATA = ROOT / "data"
STEMS = DATA / "musdb16k"
CLIPS = ROOT / "work" / "clips"
MANIFEST = ROOT / "work" / "manifest.json"

SR = 16000
CLIP_S = 10.0
SNRS = [-10, -5, 0, 5, 10]
N_UTTS = 5
N_TRACKS = 6
SEED = 20260811

CONDS = {
    "C1_rhythm": "rhythm.wav",   # drums+bass -- guaranteed vocal-free
    "C1b_accomp": "accomp.wav",  # drums+bass+other -- may carry backing vocals
    "C2_vocals": "vocals.wav",   # isolated singing
}


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def mix(speech, interf, snr_db):
    n = min(len(speech), len(interf))
    s, i = speech[:n].copy(), interf[:n].copy()
    i *= (rms(s) / (10 ** (snr_db / 20.0))) / (rms(i) + 1e-12)
    y = s + i
    pk = np.abs(y).max()
    if pk > 0.99:
        y *= 0.99 / pk
    return y.astype(np.float32)


def loudest(x, seconds=CLIP_S):
    w = int(seconds * SR)
    if len(x) <= w:
        return 0
    hop = SR // 2
    best, be = 0, -1.0
    for s0 in range(0, len(x) - w, hop):
        e = float(np.sum(x[s0:s0 + w] ** 2))
        if e > be:
            be, best = e, s0
    return best


def main():
    random.seed(SEED)
    CLIPS.mkdir(parents=True, exist_ok=True)

    trans = {}
    for tf in DATA.glob("LibriSpeech/test-clean/*/*/*.trans.txt"):
        for line in tf.read_text().splitlines():
            uid, text = line.split(" ", 1)
            trans[uid] = text

    # speech: prefer utterances close to CLIP_S so mixes are well-defined
    cands = []
    for flac in sorted(DATA.glob("LibriSpeech/test-clean/*/*/*.flac")):
        info = sf.info(str(flac))
        if 8.0 <= info.duration <= 14.0:
            cands.append(flac)
    random.shuffle(cands)
    utts = cands[:N_UTTS]

    tracks = sorted(p.name for p in STEMS.iterdir() if p.is_dir())
    random.shuffle(tracks)
    tracks = tracks[:N_TRACKS]
    print(f"{len(utts)} utts x {len(tracks)} tracks x {len(CONDS)} conds x {len(SNRS)} snrs")

    manifest = []

    def emit(name, audio, **meta):
        p = CLIPS / f"{name}.wav"
        sf.write(p, audio, SR, subtype="PCM_16")
        manifest.append({"clip": str(p), **meta})

    for flac in utts:
        uid = flac.stem
        sp, sr = sf.read(str(flac), dtype="float32")
        assert sr == SR, sr
        ref = trans[uid]
        emit(f"{uid}__speech_alone", sp, utt=uid, ref=ref, track=None,
             cond="speech_alone", snr=None)

        for tr in tracks:
            for cond, fname in CONDS.items():
                x, _ = sf.read(str(STEMS / tr / fname), dtype="float32")
                st = loudest(x)
                seg = x[st:st + int(CLIP_S * SR)]
                if len(seg) < int(CLIP_S * SR) * 0.9:
                    continue
                for snr in SNRS:
                    safe = tr.replace(" ", "_").replace("/", "_")[:40]
                    emit(f"{uid}__{safe}__{cond}__{snr:+d}", mix(sp, seg, snr),
                         utt=uid, ref=ref, track=tr, cond=cond, snr=snr)

    # singing-alone controls (no speech): what does each model do on pure song?
    for tr in tracks:
        x, _ = sf.read(str(STEMS / tr / "vocals.wav"), dtype="float32")
        st = loudest(x)
        safe = tr.replace(" ", "_").replace("/", "_")[:40]
        emit(f"CTRL__{safe}__vocals_alone", x[st:st + int(CLIP_S * SR)],
             utt=None, ref=None, track=tr, cond="vocals_alone", snr=None)

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"wrote {len(manifest)} clips -> {MANIFEST}")


if __name__ == "__main__":
    main()
