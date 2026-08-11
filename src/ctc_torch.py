"""
PyTorch re-implementation harness for whisper_ctc_offline, so its encoder can be
hooked (the deployed model is a TensorRT engine and cannot be patched).

Front-end must match the deployed DALI pipeline exactly:
  16 kHz, n_fft 400, hop 160, 128 mels, Hann, centered STFT,
  log10 -> max(x, max-8) -> (x+4)/4     [identical to Whisper's log_mel]
Bucketing: 12/24/40/60 s. A 10 s clip -> 12 s bucket -> 1200 mel frames
           -> DS6 -> 200 encoder frames (16.7 Hz).

VERIFY FIRST: transcripts must match the Triton service before any patching.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/opt/tiger/hyper_boot/models/triton/whisper_ctc_offline/build/models")
from pytorch_model_ctc_ds6 import WhisperCTCDS6Encoder  # noqa: E402

from transformers import WhisperFeatureExtractor, WhisperTokenizer  # noqa: E402

SR, HOP, NFFT, NMELS = 16000, 160, 400, 128
BLANK_ID = 50256
CKPT = "/opt/tiger/icassp2027/models/ctc/model.pt.avg10"
WHISPER_DIR = "/opt/tiger/icassp2027/models/whisper-large-v3-hf"
BUCKETS_S = [12, 24, 40, 60]

_fe = WhisperFeatureExtractor.from_pretrained(WHISPER_DIR)
_tok = WhisperTokenizer.from_pretrained(WHISPER_DIR)
_filters = torch.from_numpy(np.asarray(_fe.mel_filters, dtype=np.float32))  # (freq, mel)
if _filters.shape[0] != NMELS:
    _filters = _filters.T
_window = torch.hann_window(NFFT)


def log_mel(audio: np.ndarray, bucket_s: int) -> torch.Tensor:
    """Whisper log-mel, padded to the bucket. Returns (1, n_mels, frames)."""
    n = bucket_s * SR
    a = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    a = torch.nn.functional.pad(a, (0, max(0, n - len(a))))[:n]
    stft = torch.stft(a, NFFT, HOP, window=_window, center=True, return_complex=True)
    mag = stft[..., :-1].abs() ** 2                       # (freq, frames)
    mel = _filters.to(mag.dtype) @ mag
    log = torch.clamp(mel, min=1e-10).log10()
    log = torch.maximum(log, log.max() - 8.0)
    log = (log + 4.0) / 4.0
    return log.unsqueeze(0)                               # (1, mel, frames)


def pick_bucket(seconds: float) -> int:
    for b in BUCKETS_S:
        if seconds <= b:
            return b
    return BUCKETS_S[-1]


def load_model(device="cuda"):
    m = WhisperCTCDS6Encoder.from_pretrained(CKPT, "large-v3-turbo-fine")
    return m.eval().half().to(device)


def ctc_collapse(ids):
    out, prev = [], None
    for i in ids:
        i = int(i)
        if i != prev and i != BLANK_ID:
            out.append(i)
        prev = i
    return out


@torch.no_grad()
def transcribe(model, audio, device="cuda", real_len=None):
    secs = (real_len if real_len is not None else len(audio)) / SR
    bucket = pick_bucket(secs)
    mel = log_mel(audio, bucket).to(device, torch.float16)
    n_frames = mel.shape[-1]
    enc_len = model.conv.calculate_output_length(
        torch.tensor([n_frames], dtype=torch.int32))
    pos = torch.arange(int(enc_len[0]), dtype=torch.int32, device=device).unsqueeze(0)
    lens = torch.tensor([n_frames], dtype=torch.int32, device=device)
    yseq, lang_idx, lang_p, _ = model(mel, pos, lens)
    ids = ctc_collapse(yseq[0].tolist())
    ids = [i for i in ids if i < BLANK_ID]                # drop special/lang tokens
    return _tok.decode(ids, skip_special_tokens=True).strip()


if __name__ == "__main__":
    import soundfile as sf
    model = load_model()
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    for p in sys.argv[1:]:
        a, sr = sf.read(p, dtype="float32")
        assert sr == SR, sr
        print(f"\n{Path(p).name}\n  {transcribe(model, a)!r}")
