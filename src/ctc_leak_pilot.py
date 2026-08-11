"""
CTC lyric-leakage pilot.

Direct behavioural test of H1: whisper_ctc_offline has NO decoder, so it cannot
"choose" a source. If it transcribes lyrics anyway, the encoder retains both
sources and suppression must be a decoder behaviour.

Conditions (interferer mixed against a LibriSpeech utterance at target SNR):
  C1  accompaniment only (drums+bass+other) -- no competing voice
  C2  vocals only                           -- pure voice competition
  C3  full mixture                          -- production-like
Controls: speech alone, vocals alone.
"""
import io
import json
import re
import string
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient

ROOT = Path("/opt/tiger/icassp2027")
DATA = ROOT / "data"
ZIP = DATA / "musdb18hq.zip"
OUT = ROOT / "work" / "pilot_results.json"

HOST = "fdbd:dc53:13:72c::18"
PORT = 10823
PRE = "whisper_ctc_offline_preprocessing"
ENS = "whisper_ctc_offline_asr_ensemble"

SR = 16000
SNRS = [-10, -5, 0, 5, 10]
CLIP_S = 10.0

FEED = [
    ("AUDIO_WAVEFORM", "FP32"),
    ("MEL_INPUT_LENGTHS", "INT32"),
    ("POSITION_IDS", "INT32"),
    ("ERROR_CODE", "INT32"),
    ("TOTAL_DURATION", "FP32"),
    ("PADDING_MEL_FRAMES", "INT32"),
    ("ACTUAL_MEL_FRAMES", "INT32"),
]

client = grpcclient.InferenceServerClient(url=f"[{HOST}]:{PORT}")


# ---------------------------------------------------------------- audio utils
def to_mono16k(x, sr):
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        n = int(round(len(x) * SR / sr))
        x = np.interp(np.linspace(0, len(x), n, endpoint=False),
                      np.arange(len(x)), x).astype(np.float32)
    return x.astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def mix_at_snr(speech, interf, snr_db):
    """Scale interferer so that 10log10(P_speech/P_interf) == snr_db."""
    n = min(len(speech), len(interf))
    s, i = speech[:n], interf[:n]
    target = rms(s) / (10 ** (snr_db / 20.0))
    i = i * (target / (rms(i) + 1e-12))
    y = s + i
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def loudest_window(x, seconds=CLIP_S):
    """Pick the highest-energy window so we don't land on a silent intro."""
    w = int(seconds * SR)
    if len(x) <= w:
        return x
    hop = SR // 2
    best, best_e = 0, -1.0
    for start in range(0, len(x) - w, hop):
        e = float(np.sum(x[start:start + w] ** 2))
        if e > best_e:
            best_e, best = e, start
    return x[best:best + w]


def wav_bytes(x):
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


# ------------------------------------------------------------------ inference
def transcribe(audio):
    inp = grpcclient.InferInput("AUDIO_BYTES", audio.shape, "UINT8")
    inp.set_data_from_numpy(audio)
    pre = client.infer(PRE, [inp])
    if int(pre.as_numpy("ERROR_CODE").ravel()[0]) != 0:
        return "<preprocessing-error>"
    ens_in = []
    for name, dtype in FEED:
        arr = np.ascontiguousarray(pre.as_numpy(name))
        if name == "AUDIO_WAVEFORM":
            arr = arr.reshape(1, 1, -1).astype(np.float32)
        elif name == "POSITION_IDS":
            arr = arr.reshape(1, -1).astype(np.int32)
        else:
            arr = arr.reshape(1, 1).astype(np.float32 if dtype == "FP32" else np.int32)
        t = grpcclient.InferInput(name, arr.shape, dtype)
        t.set_data_from_numpy(arr)
        ens_in.append(t)
    out = client.infer(ENS, ens_in)
    txt = out.as_numpy("text_output")
    return txt.ravel()[0].decode() if txt is not None else ""


# -------------------------------------------------------------------- scoring
def norm_words(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in re.split(r"\s+", s) if w]


def leak_stats(hyp, ref):
    """Words emitted that are absent from the speech reference -> leak proxy."""
    h, r = norm_words(hyp), set(norm_words(ref))
    extra = [w for w in h if w not in r]
    return {
        "n_words": len(h),
        "n_extra": len(extra),
        "extra_ratio": round(len(extra) / max(len(h), 1), 3),
        "extra_sample": " ".join(extra[:12]),
    }


# ----------------------------------------------------------------------- main
def main():
    tracks = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
    z = zipfile.ZipFile(ZIP)
    all_tracks = sorted({n.split("/")[1] for n in z.namelist()
                         if n.startswith("test/") and n.count("/") > 1})
    tracks = tracks or all_tracks[:3]

    # speech utterances
    utts = []
    trans = {}
    for tf in DATA.glob("LibriSpeech/test-clean/*/*/*.trans.txt"):
        for line in tf.read_text().splitlines():
            uid, text = line.split(" ", 1)
            trans[uid] = text
    for flac in sorted(DATA.glob("LibriSpeech/test-clean/*/*/*.flac"))[:3]:
        x, sr = sf.read(str(flac))
        utts.append((flac.stem, to_mono16k(x, sr), trans[flac.stem]))
    print(f"speech utts: {[u[0] for u in utts]}", flush=True)

    results = []
    for tname in tracks:
        print(f"\n=== track: {tname} ===", flush=True)
        stems = {}
        for stem in ("vocals", "drums", "bass", "other"):
            with z.open(f"test/{tname}/{stem}.wav") as fh:
                x, sr = sf.read(io.BytesIO(fh.read()))
            stems[stem] = to_mono16k(x, sr)
        n = min(len(v) for v in stems.values())
        vocals = stems["vocals"][:n]
        accomp = (stems["drums"][:n] + stems["bass"][:n] + stems["other"][:n])
        mixture = vocals + accomp

        # align windows on the vocal-loudest region so singing is present
        w = int(CLIP_S * SR)
        hop = SR // 2
        best, best_e = 0, -1.0
        for s0 in range(0, max(len(vocals) - w, 1), hop):
            e = float(np.sum(vocals[s0:s0 + w] ** 2))
            if e > best_e:
                best_e, best = e, s0
        seg = slice(best, best + w)
        interferers = {"C1_accomp": accomp[seg], "C2_vocals": vocals[seg],
                       "C3_mixture": mixture[seg]}

        # control: what does the model do on singing alone?
        soloc = transcribe(wav_bytes(interferers["C2_vocals"]))
        print(f"  [vocals-alone] {soloc!r}", flush=True)
        results.append({"track": tname, "cond": "vocals_alone", "snr": None,
                        "utt": None, "hyp": soloc})

        for uid, sp, ref in utts:
            base = transcribe(wav_bytes(sp))
            results.append({"track": tname, "cond": "speech_alone", "snr": None,
                            "utt": uid, "hyp": base, "ref": ref,
                            **leak_stats(base, ref)})
            for cname, interf in interferers.items():
                for snr in SNRS:
                    y = mix_at_snr(sp, interf, snr)
                    hyp = transcribe(wav_bytes(y))
                    st = leak_stats(hyp, ref)
                    results.append({"track": tname, "cond": cname, "snr": snr,
                                    "utt": uid, "hyp": hyp, "ref": ref, **st})
                    print(f"  {cname:11s} SNR{snr:+4d}  extra={st['n_extra']:3d}"
                          f"/{st['n_words']:3d}  {hyp[:88]!r}", flush=True)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\nwrote {OUT} ({len(results)} rows)")


if __name__ == "__main__":
    main()
