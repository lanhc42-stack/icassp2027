"""Smoke test for whisper_ctc_offline: preprocessing -> asr_ensemble."""
import sys
import numpy as np
import tritonclient.grpc as grpcclient

URL = "fdbd:dc53:13:72c::18:10823"
PRE = "whisper_ctc_offline_preprocessing"
ENS = "whisper_ctc_offline_asr_ensemble"

# asr_ensemble inputs and the preprocessing outputs that feed them
FEED = [
    ("AUDIO_WAVEFORM", "FP32"),
    ("MEL_INPUT_LENGTHS", "INT32"),
    ("POSITION_IDS", "INT32"),
    ("ERROR_CODE", "INT32"),
    ("TOTAL_DURATION", "FP32"),
    ("PADDING_MEL_FRAMES", "INT32"),
    ("ACTUAL_MEL_FRAMES", "INT32"),
]


def main(path):
    audio = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    client = grpcclient.InferenceServerClient(url=f"[{URL.rsplit(':', 1)[0]}]:{URL.rsplit(':', 1)[1]}")

    # --- stage 1: preprocessing (max_batch_size 0 -> no batch dim) ---
    inp = grpcclient.InferInput("AUDIO_BYTES", audio.shape, "UINT8")
    inp.set_data_from_numpy(audio)
    pre = client.infer(PRE, [inp])

    err = pre.as_numpy("ERROR_CODE")
    print(f"preprocessing ERROR_CODE={err.ravel()[0]}")
    for n in ("TOTAL_DURATION", "ACTUAL_MEL_FRAMES", "LANGUAGE_RESULT"):
        v = pre.as_numpy(n)
        if v is not None:
            print(f"  {n}: {v.ravel()[:3]}")

    # --- stage 2: asr_ensemble (max_batch_size 4 -> prepend batch dim) ---
    ens_inputs = []
    for name, dtype in FEED:
        arr = pre.as_numpy(name)
        if arr is None:
            print(f"MISSING preprocessing output: {name}")
            return 1
        arr = np.ascontiguousarray(arr)
        if name == "AUDIO_WAVEFORM":
            arr = arr.reshape(1, 1, -1).astype(np.float32)
        elif name == "POSITION_IDS":
            arr = arr.reshape(1, -1).astype(np.int32)
        else:
            arr = arr.reshape(1, 1).astype(np.float32 if dtype == "FP32" else np.int32)
        t = grpcclient.InferInput(name, arr.shape, dtype)
        t.set_data_from_numpy(arr)
        ens_inputs.append(t)
        print(f"  feed {name} {arr.shape} {arr.dtype}")

    out = client.infer(ENS, ens_inputs)
    text = out.as_numpy("text_output")
    lang = out.as_numpy("detected_language")
    conf = out.as_numpy("language_confidence")
    print("\n=== RESULT ===")
    print("text:", text.ravel()[0].decode() if text is not None else None)
    print("lang:", lang.ravel()[0].decode() if lang is not None else None)
    print("conf:", conf.ravel()[0] if conf is not None else None)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
