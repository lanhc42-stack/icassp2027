"""
Extract MUSDB18-HQ test stems as 16 kHz mono wavs, so the 22.7 GB archive can
be deleted. Uses torchaudio's proper resampler (the pilot used linear interp,
which aliases -- fine for a smoke test, not for reported numbers).

Writes: data/musdb16k/<track>/{vocals.wav,rhythm.wav}
"""
import io
import zipfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

ROOT = Path("/opt/tiger/icassp2027/data")
ZIP = ROOT / "musdb18hq.zip"
OUT = ROOT / "musdb16k"
SR = 16000

z = zipfile.ZipFile(ZIP)
tracks = sorted({n.split("/")[1] for n in z.namelist()
                 if n.startswith("test/") and n.count("/") > 1})
print(f"{len(tracks)} test tracks", flush=True)

resamplers = {}


def load16k(name):
    with z.open(name) as fh:
        x, sr = sf.read(io.BytesIO(fh.read()), dtype="float32")
    t = torch.from_numpy(x)
    if t.ndim > 1:
        t = t.mean(dim=1)
    if sr != SR:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(sr, SR)
        t = resamplers[sr](t)
    return t


for i, tr in enumerate(tracks, 1):
    d = OUT / tr
    if (d / "rhythm.wav").exists():
        continue
    d.mkdir(parents=True, exist_ok=True)
    voc = load16k(f"test/{tr}/vocals.wav")
    acc = None
    for stem in ("drums", "bass"):
        s = load16k(f"test/{tr}/{stem}.wav")
        acc = s if acc is None else acc[:len(s)] + s[:len(acc)]
    n = min(len(voc), len(acc))
    
    sf.write(d / "rhythm.wav", acc[:n].numpy(), SR, subtype="PCM_16")
    print(f"[{i}/{len(tracks)}] {tr}  {n/SR:.0f}s", flush=True)

print("done", flush=True)
