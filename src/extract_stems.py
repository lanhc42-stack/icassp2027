"""Prepare MUSDB18 vocals and accompaniment as 16 kHz mono WAV files.

Both public MUSDB18 layouts are supported:

* compressed MUSDB18: ``<root>/<subset>/*.stem.mp4``;
* MUSDB18-HQ: ``<root>/<subset>/<track>/{vocals,drums,bass,other}.wav``.

The experiment code only consumes ``vocals.wav``.  ``accomp.wav`` is emitted as
well so that later oracle/source-separation experiments can reuse the same
prepared tracks.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _mono_resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim == 2:
        value = value.mean(axis=1)
    if value.ndim != 1:
        raise ValueError(f"expected mono/stereo audio, got shape={value.shape}")
    if source_rate != target_rate:
        divisor = math.gcd(source_rate, target_rate)
        value = resample_poly(
            value,
            target_rate // divisor,
            source_rate // divisor,
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(value)


def _write_track(
    destination: Path,
    vocals: np.ndarray,
    accompaniment: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    vocal_mono = _mono_resample(vocals, source_rate, target_rate)
    accomp_mono = _mono_resample(accompaniment, source_rate, target_rate)
    length = min(len(vocal_mono), len(accomp_mono))
    sf.write(
        destination / "vocals.wav",
        np.clip(vocal_mono[:length], -1.0, 1.0),
        target_rate,
        subtype="PCM_16",
    )
    sf.write(
        destination / "accomp.wav",
        np.clip(accomp_mono[:length], -1.0, 1.0),
        target_rate,
        subtype="PCM_16",
    )


def _compressed_tracks(root: Path, subset: str) -> list[Path]:
    return sorted((root / subset).glob("*.stem.mp4"))


def _hq_tracks(root: Path, subset: str) -> list[Path]:
    return sorted((root / subset).glob("*/vocals.wav"))


def prepare(
    root: Path,
    output: Path,
    *,
    subset: str,
    sample_rate: int,
    limit: int | None,
    overwrite: bool,
) -> None:
    compressed = _compressed_tracks(root, subset)
    hq = _hq_tracks(root, subset)
    if compressed:
        import stempeg

        sources = compressed[:limit] if limit else compressed
        print(f"found {len(compressed)} compressed {subset} tracks", flush=True)
        for index, source in enumerate(sources, 1):
            track = source.name.removesuffix(".stem.mp4")
            destination = output / track
            if not overwrite and (destination / "vocals.wav").exists() and (
                destination / "accomp.wav"
            ).exists():
                continue
            # MUSDB STEMS order: mixture, drums, bass, other, vocals.
            stems, source_rate = stempeg.read_stems(
                str(source), stem_id=[1, 2, 3, 4], dtype=np.float32
            )
            if stems.shape[0] != 4:
                raise RuntimeError(f"unexpected stem shape for {source}: {stems.shape}")
            drums, bass, other, vocals = stems
            accompaniment = drums + bass + other
            _write_track(
                destination,
                vocals,
                accompaniment,
                int(source_rate),
                sample_rate,
            )
            print(f"[{index}/{len(sources)}] {track}", flush=True)
        return

    if hq:
        sources = hq[:limit] if limit else hq
        print(f"found {len(hq)} HQ {subset} tracks", flush=True)
        for index, vocal_path in enumerate(sources, 1):
            track = vocal_path.parent.name
            destination = output / track
            if not overwrite and (destination / "vocals.wav").exists() and (
                destination / "accomp.wav"
            ).exists():
                continue
            vocals, source_rate = sf.read(vocal_path, dtype="float32", always_2d=True)
            accompaniment = np.zeros_like(vocals)
            for stem_name in ("drums.wav", "bass.wav", "other.wav"):
                stem, stem_rate = sf.read(
                    vocal_path.parent / stem_name, dtype="float32", always_2d=True
                )
                if stem_rate != source_rate:
                    raise ValueError(f"sample-rate mismatch in {vocal_path.parent}")
                length = min(len(accompaniment), len(stem))
                accompaniment = accompaniment[:length] + stem[:length]
                vocals = vocals[:length]
            _write_track(
                destination,
                vocals,
                accompaniment,
                int(source_rate),
                sample_rate,
            )
            print(f"[{index}/{len(sources)}] {track}", flush=True)
        return

    raise RuntimeError(
        f"no MUSDB tracks found below {root / subset}; expected *.stem.mp4 or */vocals.wav"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset", default="test")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        subset=args.subset,
        sample_rate=args.sample_rate,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
