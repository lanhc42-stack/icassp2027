from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import yaml
from scipy.signal import stft

from .audio import common_scale, peak, read_audio, read_pcm16, rms, vocal_gain_for_snr, write_pcm16
from .e4 import WhisperPromptRunner
from .records import append_jsonl, read_jsonl, sha256_file, stable_id, write_jsonl
from .scoring import attribute_tokens, normalize_words


VARIANTS = ("intact", "shuffle_500ms", "shuffle_100ms", "reverse", "instrumental")


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("E5 config must be a mapping")
    return config


def prepare(config: dict[str, Any]) -> Path:
    source = Path(config["project"]["pairs_manifest"])
    rows = [
        row for row in read_jsonl(source)
        if row.get("split") == config["project"].get("split", "development")
    ]
    selected = _select_pairs(rows, int(config["selection"]["crops_per_track"]))
    output_root = Path(config["project"]["output_root"])
    audio_root = output_root / "audio"
    variant_root = output_root / "variants"
    sample_rate = int(config["audio"]["sample_rate"])
    duration = float(config["audio"]["clip_duration_s"])
    expected_samples = round(sample_rate * duration)
    seed = int(config["project"]["seed"])
    snr_db = float(config["audio"]["snr_db"])
    headroom = float(config["audio"]["headroom_peak"])
    smoothing_ms = float(config["audio"]["boundary_smoothing_ms"])

    by_crop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_crop[row["vocal_crop_id"]].append(row)
    manifest: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for crop_id, pair_rows in sorted(by_crop.items()):
        exemplar = pair_rows[0]
        vocal, vocal_sr = read_pcm16(exemplar["vocal_path"])
        if vocal_sr != sample_rate or len(vocal) != expected_samples:
            raise ValueError(f"unexpected prepared vocal shape for {crop_id}")
        accompaniment = _load_accompaniment(exemplar, sample_rate, expected_samples)
        variants = _variants(vocal, accompaniment, sample_rate, smoothing_ms, seed, crop_id)
        solo_scale = common_scale(list(variants.values()), headroom)
        variant_paths: dict[str, Path] = {}
        for name, audio in variants.items():
            path = variant_root / crop_id / f"{name}.wav"
            write_pcm16(path, audio * solo_scale, sample_rate)
            variant_paths[name] = path.resolve()
            checks.append(_audio_check(crop_id, name, vocal, audio, sample_rate))
            manifest.append(
                {
                    **_reference_fields(exemplar),
                    "experiment": "e5",
                    "mode": "solo",
                    "variant": name,
                    "sample_id": stable_id("e5", "solo", crop_id, name),
                    "audio_path": str(path.resolve()),
                    "audio_sha256": sha256_file(path),
                    "snr_db": None,
                    "common_scale": solo_scale,
                }
            )
        for pair in pair_rows:
            speech, speech_sr = read_pcm16(pair["speech_path"])
            if speech_sr != sample_rate or len(speech) != expected_samples:
                raise ValueError(f"unexpected prepared speech shape for {pair['pair_id']}")
            mixtures: dict[str, np.ndarray] = {}
            for name, background in variants.items():
                gain = vocal_gain_for_snr(speech, background, snr_db)
                mixtures[name] = speech + gain * background
            mix_scale = common_scale(list(mixtures.values()), headroom)
            for name, mixture in mixtures.items():
                sample_id = stable_id("e5", "mix", pair["pair_id"], name)
                path = audio_root / f"{sample_id}.wav"
                write_pcm16(path, mixture * mix_scale, sample_rate)
                manifest.append(
                    {
                        **_reference_fields(pair),
                        "experiment": "e5",
                        "mode": "mix",
                        "variant": name,
                        "sample_id": sample_id,
                        "audio_path": str(path.resolve()),
                        "audio_sha256": sha256_file(path),
                        "snr_db": snr_db,
                        "common_scale": mix_scale,
                    }
                )
    destination = output_root / "manifest.jsonl"
    write_jsonl(destination, manifest)
    (output_root / "manipulation_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def _select_pairs(rows: list[dict[str, Any]], crops_per_track: int) -> list[dict[str, Any]]:
    by_track_crop: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_track_crop[str(row["track_id"])][str(row["vocal_crop_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for track, crops in sorted(by_track_crop.items()):
        ranked = sorted(crops.items(), key=lambda item: stable_id("e5", track, item[0]))
        for _, pair_rows in ranked[:crops_per_track]:
            selected.extend(sorted(pair_rows, key=lambda row: row["speech_id"]))
    return selected


def _load_accompaniment(row: dict[str, Any], sample_rate: int, width: int) -> np.ndarray:
    directory = Path(row["vocal_source_path"]).parent
    candidates = (directory / "accomp.wav", directory / "accompaniment.wav")
    source = next((path for path in candidates if path.exists()), candidates[0])
    audio = read_audio(source, sample_rate)
    start = round(float(row["vocal_crop_start_s"]) * sample_rate)
    crop = audio[start : start + width]
    if len(crop) != width:
        raise ValueError(f"short accompaniment crop: {source}")
    return crop


def _variants(
    vocal: np.ndarray,
    accompaniment: np.ndarray,
    sample_rate: int,
    smoothing_ms: float,
    seed: int,
    crop_id: str,
) -> dict[str, np.ndarray]:
    values = {
        "intact": vocal.copy(),
        "shuffle_500ms": _block_shuffle(vocal, round(0.5 * sample_rate), smoothing_ms, sample_rate, seed, crop_id),
        "shuffle_100ms": _block_shuffle(vocal, round(0.1 * sample_rate), smoothing_ms, sample_rate, seed + 1, crop_id),
        "reverse": vocal[::-1].copy(),
        "instrumental": _spectral_match(accompaniment, vocal),
    }
    target_rms = rms(vocal)
    for name, audio in values.items():
        values[name] = (audio * (target_rms / rms(audio))).astype(np.float32)
    return values


def _block_shuffle(
    audio: np.ndarray,
    block: int,
    smoothing_ms: float,
    sample_rate: int,
    seed: int,
    crop_id: str,
) -> np.ndarray:
    count = len(audio) // block
    blocks = [audio[index * block : (index + 1) * block] for index in range(count)]
    tail = audio[count * block :]
    local_seed = int(stable_id(seed, crop_id, block), 16) % (2**32)
    rng = np.random.default_rng(local_seed)
    order = rng.permutation(count)
    if np.all(order == np.arange(count)) and count > 1:
        order = np.roll(order, 1)
    output = np.concatenate([*(blocks[index] for index in order), tail]).astype(np.float32)
    radius = max(1, round(smoothing_ms * sample_rate / 2000.0))
    for boundary in range(block, count * block, block):
        left, right = boundary - radius, boundary + radius
        if left < 0 or right > len(output):
            continue
        output[left:right] = np.linspace(output[left], output[right - 1], right - left, dtype=np.float32)
    return output


def _spectral_match(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_spectrum = np.fft.rfft(source.astype(np.float64))
    target_magnitude = np.abs(np.fft.rfft(target.astype(np.float64)))
    phase = np.exp(1j * np.angle(source_spectrum))
    return np.fft.irfft(target_magnitude * phase, n=len(target)).astype(np.float32)


def _audio_check(crop_id: str, name: str, intact: np.ndarray, variant: np.ndarray, sample_rate: int) -> dict[str, Any]:
    _, _, intact_stft = stft(intact, fs=sample_rate, nperseg=400, noverlap=240)
    _, _, variant_stft = stft(variant, fs=sample_rate, nperseg=400, noverlap=240)
    a = np.mean(np.abs(intact_stft), axis=1) + 1e-9
    b = np.mean(np.abs(variant_stft), axis=1) + 1e-9
    spectral_db_mae = float(np.mean(np.abs(20 * np.log10(a) - 20 * np.log10(b))))
    modulation_distance = _modulation_distance(intact, variant, sample_rate)
    return {
        "vocal_crop_id": crop_id,
        "variant": name,
        "rms": rms(variant),
        "peak": peak(variant),
        "long_term_spectral_db_mae_from_intact": spectral_db_mae,
        "amplitude_modulation_cosine_distance_from_intact": modulation_distance,
    }


def _modulation_distance(a: np.ndarray, b: np.ndarray, sample_rate: int) -> float:
    frame, hop = round(0.025 * sample_rate), round(0.01 * sample_rate)
    def envelope(x: np.ndarray) -> np.ndarray:
        starts = range(0, len(x) - frame + 1, hop)
        return np.asarray([rms(x[start : start + frame]) for start in starts])
    x, y = np.abs(np.fft.rfft(envelope(a))), np.abs(np.fft.rfft(envelope(b)))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 1.0 - float(np.dot(x, y) / denominator) if denominator else 0.0


def _reference_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "pair_id", "track_id", "vocal_crop_id", "speech_id", "speech_speaker_id",
            "speech_reference", "lyric_reference", "lyric_reference_scope", "split",
        )
    }


def run(config: dict[str, Any], *, limit: int | None = None) -> Path:
    output_root = Path(config["project"]["output_root"])
    manifest = output_root / "manifest.jsonl"
    if not manifest.exists():
        prepare(config)
    destination = output_root / "runs" / "whisper.jsonl"
    completed = {row["sample_id"] for row in read_jsonl(destination)} if destination.exists() else set()
    rows = [row for row in read_jsonl(manifest) if row["sample_id"] not in completed]
    if limit is not None:
        rows = rows[:limit]
    runner = WhisperPromptRunner(config["model"])
    started = time.time()
    for index, row in enumerate(rows, 1):
        audio, sample_rate = read_pcm16(row["audio_path"])
        try:
            result = runner.transcribe(audio, sample_rate, "")
            result.update(error=None)
        except Exception as exc:
            result = {"hyp": "", "error": f"{type(exc).__name__}: {exc}"}
        score = _score(row, result, config["scoring"])
        append_jsonl(destination, {**row, **result, **score, "model": "whisper"})
        if index % 20 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] {time.time() - started:.1f}s", flush=True)
    return destination


def _score(row: dict[str, Any], result: dict[str, Any], scoring: dict[str, Any]) -> dict[str, Any]:
    if result.get("error"):
        return {"lir": None, "tsr": None, "scs": None, "lyric_recall": None, "score_error": "inference error"}
    if row["mode"] == "solo":
        hypothesis = normalize_words(result["hyp"], scoring)
        lyric = normalize_words(row["lyric_reference"], scoring)
        matched = attribute_tokens(result["hyp"], "", row["lyric_reference"], scoring)
        return {
            **matched,
            "lyric_recall": matched["n_lyric"] / len(lyric) if lyric else None,
            "hypothesis_word_count": len(hypothesis),
            "score_error": None,
        }
    score = attribute_tokens(result["hyp"], row["speech_reference"], row["lyric_reference"], scoring)
    score.update(lyric_recall=None, score_error=None)
    return score


def analyze(config: dict[str, Any]) -> Path:
    root = Path(config["project"]["output_root"])
    rows = [row for row in read_jsonl(root / "runs" / "whisper.jsonl") if row.get("score_error") is None]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["variant"])].append(row)
    aggregates = []
    for (mode, variant), values in sorted(grouped.items()):
        record = {"mode": mode, "variant": variant, "n": len(values), "n_tracks": len({r["track_id"] for r in values})}
        metrics = ("lyric_recall",) if mode == "solo" else ("lir", "tsr", "scs")
        for metric in metrics:
            available = [float(row[metric]) for row in values if row.get(metric) is not None]
            record[f"mean_{metric}"] = mean(available) if available else None
        record["mean_no_grounded_output"] = mean(
            float(bool(row.get("no_grounded_output"))) for row in values
        ) if values else None
        aggregates.append(record)
    contrasts = _paired_contrasts(rows, config)
    association = _solo_mix_association(rows)
    summary = {"experiment": "e5", "n_rows": len(rows), "aggregates": aggregates, "contrasts": contrasts, "solo_mix_association": association}
    destination = root / "summary.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _paired_contrasts(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    mixes = [row for row in rows if row["mode"] == "mix"]
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in mixes:
        by_pair[row["pair_id"]][row["variant"]] = row
    output = []
    for variant in VARIANTS[1:]:
        for metric in ("lir", "tsr", "scs"):
            effects: list[tuple[str, float]] = []
            for conditions in by_pair.values():
                if "intact" not in conditions or variant not in conditions:
                    continue
                a, b = conditions[variant].get(metric), conditions["intact"].get(metric)
                if a is not None and b is not None:
                    effects.append((conditions[variant]["track_id"], float(a) - float(b)))
            output.append(_bootstrap_record(variant, metric, effects, config))
    return output


def _bootstrap_record(variant: str, metric: str, effects: list[tuple[str, float]], config: dict[str, Any]) -> dict[str, Any]:
    by_track: dict[str, list[float]] = defaultdict(list)
    for track, value in effects:
        by_track[track].append(value)
    values = np.asarray([mean(items) for items in by_track.values()], dtype=np.float64)
    estimate = low = high = None
    if len(values):
        rng = np.random.default_rng(int(config["project"]["seed"]))
        draws = rng.choice(values, size=(int(config["scoring"]["bootstrap_replicates"]), len(values)), replace=True).mean(axis=1)
        estimate, low, high = float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    return {"contrast": f"{metric}({variant})-{metric}(intact)", "estimate": estimate, "ci95_low": low, "ci95_high": high, "n_pairs": len(effects), "n_tracks": len(values)}


def _solo_mix_association(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solo = {(row["vocal_crop_id"], row["variant"]): row.get("lyric_recall") for row in rows if row["mode"] == "solo"}
    points = []
    for row in rows:
        if row["mode"] != "mix" or row.get("lir") is None:
            continue
        recall = solo.get((row["vocal_crop_id"], row["variant"]))
        if recall is not None:
            points.append((float(recall), float(row["lir"])))
    if len(points) < 2:
        return {"n": len(points), "spearman_solo_recall_vs_mix_lir": None}
    from scipy.stats import spearmanr
    statistic = spearmanr([x for x, _ in points], [y for _, y in points]).statistic
    return {"n": len(points), "spearman_solo_recall_vs_mix_lir": float(statistic)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E5 lexical-intelligibility intervention")
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    path = prepare(config) if args.command == "prepare" else run(config, limit=args.limit) if args.command == "run" else analyze(config)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
