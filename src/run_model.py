"""
Transcribe the fixed mixture manifest with one model on the prior-strength
ladder. Every backend sees byte-identical wavs.

  python3 run_model.py ctc
  python3 run_model.py whisper /opt/tiger/icassp2027/models/whisper-large-v3

Language is auto-detected (never forced) so the ladder is comparable and so
language-flipping under heavy music stays observable.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import os

ROOT = Path("/opt/tiger/icassp2027")
MANIFEST = Path(os.environ.get("MANIFEST_PATH", ROOT / "work" / "manifest.json"))
TAG_SUFFIX = os.environ.get("TAG_SUFFIX", "")
OUTDIR = ROOT / "work" / "runs"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ backends
def backend_ctc():
    import tritonclient.grpc as grpcclient
    HOST, PORT = "fdbd:dc53:13:72c::18", 10823
    PRE = "whisper_ctc_offline_preprocessing"
    ENS = "whisper_ctc_offline_asr_ensemble"
    FEED = [("AUDIO_WAVEFORM", "FP32"), ("MEL_INPUT_LENGTHS", "INT32"),
            ("POSITION_IDS", "INT32"), ("ERROR_CODE", "INT32"),
            ("TOTAL_DURATION", "FP32"), ("PADDING_MEL_FRAMES", "INT32"),
            ("ACTUAL_MEL_FRAMES", "INT32")]
    cli = grpcclient.InferenceServerClient(url=f"[{HOST}]:{PORT}")

    def run(path):
        raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        t = grpcclient.InferInput("AUDIO_BYTES", raw.shape, "UINT8")
        t.set_data_from_numpy(raw)
        pre = cli.infer(PRE, [t])
        if int(pre.as_numpy("ERROR_CODE").ravel()[0]) != 0:
            return "", "ERR"
        ins = []
        for name, dt in FEED:
            a = np.ascontiguousarray(pre.as_numpy(name))
            if name == "AUDIO_WAVEFORM":
                a = a.reshape(1, 1, -1).astype(np.float32)
            elif name == "POSITION_IDS":
                a = a.reshape(1, -1).astype(np.int32)
            else:
                a = a.reshape(1, 1).astype(np.float32 if dt == "FP32" else np.int32)
            i = grpcclient.InferInput(name, a.shape, dt)
            i.set_data_from_numpy(a)
            ins.append(i)
        out = cli.infer(ENS, ins)
        txt = out.as_numpy("text_output")
        lang = out.as_numpy("detected_language")
        return (txt.ravel()[0].decode() if txt is not None else "",
                lang.ravel()[0].decode() if lang is not None else "")
    return run


def backend_whisper(model_dir):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float16).to("cuda").eval()

    def run(path):
        x, sr = sf.read(path, dtype="float32")
        feats = proc(x, sampling_rate=sr, return_tensors="pt").input_features
        feats = feats.to("cuda", torch.float16)
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=180)
        text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        # recover the language token the model chose
        toks = proc.batch_decode(ids, skip_special_tokens=False)[0]
        lang = ""
        for t in toks.split("<|"):
            t = t.split("|>")[0]
            if len(t) == 2 and t.isalpha():
                lang = t
                break
        return text, lang
    return run


# ---------------------------------------------------------------------- main
def main():
    which = sys.argv[1]
    if which == "ctc":
        run, tag = backend_ctc(), "ctc"
    elif which == "whisper":
        run, tag = backend_whisper(sys.argv[2]), "whisper_large_v3"
    else:
        raise SystemExit(f"unknown backend {which}")

    rows = json.loads(MANIFEST.read_text())
    out = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        try:
            text, lang = run(r["clip"])
        except Exception as e:
            text, lang = f"<error: {type(e).__name__}: {e}>", ""
        out.append({**{k: v for k, v in r.items() if k != "clip"},
                    "hyp": text, "lang": lang})
        if i % 50 == 0 or i == len(rows):
            el = time.time() - t0
            print(f"  [{i}/{len(rows)}] {el:.0f}s  ({el/i:.2f}s/clip)", flush=True)

    p = OUTDIR / f"{tag}{TAG_SUFFIX}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
