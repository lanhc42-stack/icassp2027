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

from .audio import read_pcm16
from .records import append_jsonl, read_jsonl, stable_id, write_jsonl
from .scoring import attribute_tokens


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("E6 config must be a mapping")
    return value


def prepare(config: dict[str, Any]) -> Path:
    score_path = Path(config["project"]["e1_root"]) / "scores" / "whisper.jsonl"
    rows = [row for row in read_jsonl(score_path) if row.get("split") == "development"]
    lookup = {(row["pair_id"], float(row["snr_db"]), row["condition"]): row for row in rows}
    settings = config["selection"]
    snr = float(settings["snr_db"])
    eligible: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for (pair_id, row_snr, condition), lyric in lookup.items():
        if row_snr != snr or condition != "baseline" or lyric.get("scs") is None:
            continue
        speech = lookup.get((pair_id, snr, "full"))
        if not speech or speech.get("scs") is None:
            continue
        if (
            float(lyric["scs"]) >= float(settings["lyric_receiver_scs_min"])
            and float(speech["scs"]) <= float(settings["speech_receiver_scs_max"])
        ):
            eligible.append((lyric, speech))
    by_track: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in eligible:
        by_track[str(pair[0]["track_id"])].append(pair)
    selected = []
    for track, pairs in sorted(by_track.items()):
        ordered = sorted(pairs, key=lambda pair: stable_id("e6", track, pair[0]["pair_id"]))
        selected.extend(ordered[: int(settings["pairs_per_track"])])
    manifest = []
    for lyric, speech in selected:
        manifest.append(
            {
                "patch_pair_id": stable_id("e6", lyric["pair_id"]),
                "pair_id": lyric["pair_id"],
                "track_id": lyric["track_id"],
                "speech_id": lyric["speech_id"],
                "speech_speaker_id": lyric["speech_speaker_id"],
                "split": lyric["split"],
                "speech_reference": lyric["speech_reference"],
                "lyric_reference": lyric["lyric_reference"],
                "lyric_audio_path": lyric["audio_path"],
                "speech_audio_path": speech["audio_path"],
                "selection_lyric_scs": lyric["scs"],
                "selection_speech_scs": speech["scs"],
            }
        )
    if len(manifest) < 20:
        raise RuntimeError(f"E6 requires at least 20 independent tracks, found {len(manifest)}")
    destination = Path(config["project"]["output_root"]) / "manifest.jsonl"
    write_jsonl(destination, manifest)
    return destination


class EncoderPatcher:
    def __init__(self, config: dict[str, Any]):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.device = str(config.get("device", "cuda"))
        self.dtype = getattr(torch, str(config.get("dtype", "float16")))
        self.config = config
        self.processor = WhisperProcessor.from_pretrained(config["model_path"])
        self.model = WhisperForConditionalGeneration.from_pretrained(
            config["model_path"], torch_dtype=self.dtype
        ).to(self.device).eval()
        self.cache: dict[int, Any] = {}
        self.plan: tuple[int, int, int] | None = None
        self.handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(self.model.model.encoder.layers)
        ]

    def _hook(self, index: int):
        def hook(module: Any, args: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            if self.plan is None:
                self.cache[index] = hidden.detach().clone()
                return output
            layer, start, end = self.plan
            if index != layer or index not in self.cache:
                return output
            patched = hidden.clone()
            patched[:, start:end, :] = self.cache[index][:, start:end, :].to(hidden.dtype)
            if isinstance(output, tuple):
                return (patched,) + tuple(output[1:])
            return patched
        return hook

    def features(self, audio: np.ndarray, sample_rate: int) -> Any:
        values = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        return values.input_features.to(device=self.device, dtype=self.dtype)

    def record(self, features: Any) -> tuple[str, dict[int, Any]]:
        self.plan = None
        self.cache.clear()
        text = self.decode(features)
        return text, {index: value.clone() for index, value in self.cache.items()}

    def patched(self, features: Any, snapshot: dict[int, Any], layer: int, start: int, end: int) -> str:
        self.cache.clear()
        self.cache.update(snapshot)
        self.plan = (layer, start, end)
        try:
            return self.decode(features)
        finally:
            self.plan = None

    def decode(self, features: Any) -> str:
        kwargs = {
            "language": self.config.get("language", "en"),
            "task": self.config.get("task", "transcribe"),
            "max_new_tokens": int(self.config.get("max_new_tokens", 180)),
            "return_timestamps": False,
        }
        with self.torch.inference_mode():
            tokens = self.model.generate(features, **kwargs)
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def run(config: dict[str, Any], *, limit_pairs: int | None = None) -> Path:
    root = Path(config["project"]["output_root"])
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        prepare(config)
    rows = list(read_jsonl(manifest))
    if limit_pairs is not None:
        rows = rows[:limit_pairs]
    destination = root / "runs" / "whisper.jsonl"
    completed = {row["patch_id"] for row in read_jsonl(destination)} if destination.exists() else set()
    patcher = EncoderPatcher(config["model"])
    patching = config["patching"]
    layers = [int(value) for value in patching["layers"]]
    windows = {name: (int(value[0]), int(value[1])) for name, value in patching["windows"].items()}
    real_frames = int(patching["real_frames"])
    self_layer = int(patching["self_control_layer"])
    started = time.time()
    for pair_index, row in enumerate(rows, 1):
        lyric_audio, lyric_sr = read_pcm16(row["lyric_audio_path"])
        speech_audio, speech_sr = read_pcm16(row["speech_audio_path"])
        if lyric_sr != speech_sr:
            raise ValueError("E6 donor/receiver sample-rate mismatch")
        lyric_features = patcher.features(lyric_audio, lyric_sr)
        speech_features = patcher.features(speech_audio, speech_sr)
        lyric_text, lyric_snapshot = patcher.record(lyric_features)
        speech_text, speech_snapshot = patcher.record(speech_features)
        base = {
            "lyric": _score(lyric_text, row, config),
            "speech": _score(speech_text, row, config),
        }
        directions = (
            ("speech_to_lyric", lyric_features, speech_snapshot, lyric_snapshot, "lyric", "speech"),
            ("lyric_to_speech", speech_features, lyric_snapshot, speech_snapshot, "speech", "lyric"),
        )
        for direction, receiver_features, donor_snapshot, receiver_snapshot, receiver_name, donor_name in directions:
            for layer in layers:
                patch_id = stable_id(row["patch_pair_id"], direction, layer, "full_real")
                if patch_id not in completed:
                    text = patcher.patched(receiver_features, donor_snapshot, layer, 0, real_frames)
                    _write_patch(destination, row, patch_id, direction, layer, "full_real", 0, real_frames, text, base, receiver_name, donor_name, config)
                for position, (start, end) in windows.items():
                    patch_id = stable_id(row["patch_pair_id"], direction, layer, position)
                    if patch_id in completed:
                        continue
                    text = patcher.patched(receiver_features, donor_snapshot, layer, start, end)
                    _write_patch(destination, row, patch_id, direction, layer, position, start, end, text, base, receiver_name, donor_name, config)
            for position, (start, end) in windows.items():
                patch_id = stable_id(row["patch_pair_id"], direction, "self", self_layer, position)
                if patch_id in completed:
                    continue
                text = patcher.patched(receiver_features, receiver_snapshot, self_layer, start, end)
                _write_patch(destination, row, patch_id, direction, self_layer, f"self_{position}", start, end, text, base, receiver_name, donor_name, config)
        print(f"[{pair_index}/{len(rows)}] {time.time() - started:.1f}s", flush=True)
    return destination


def _score(text: str, row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return attribute_tokens(text, row["speech_reference"], row["lyric_reference"], config["scoring"])


def _write_patch(
    destination: Path,
    row: dict[str, Any],
    patch_id: str,
    direction: str,
    layer: int,
    position: str,
    start: int,
    end: int,
    text: str,
    base: dict[str, dict[str, Any]],
    receiver_name: str,
    donor_name: str,
    config: dict[str, Any],
) -> None:
    score = _score(text, row, config)
    append_jsonl(
        destination,
        {
            **row,
            "experiment": "e6",
            "model": "whisper",
            "patch_id": patch_id,
            "direction": direction,
            "layer": layer,
            "position": position,
            "frame_start": start,
            "frame_end": end,
            "hyp": text,
            **score,
            "receiver_name": receiver_name,
            "donor_name": donor_name,
            "receiver_hyp": base[receiver_name].get("normalized_hypothesis"),
            "receiver_scs": base[receiver_name].get("scs"),
            "donor_scs": base[donor_name].get("scs"),
            "error": None,
        },
    )


def analyze(config: dict[str, Any]) -> Path:
    root = Path(config["project"]["output_root"])
    rows = list(read_jsonl(root / "runs" / "whisper.jsonl"))
    full = {
        (row["patch_pair_id"], row["direction"], int(row["layer"])): row
        for row in rows if row["position"] == "full_real"
    }
    normalized = []
    for row in rows:
        if row["position"] not in config["patching"]["windows"] or row.get("scs") is None or row.get("receiver_scs") is None:
            continue
        comparator = full.get((row["patch_pair_id"], row["direction"], int(row["layer"])))
        if not comparator or comparator.get("scs") is None:
            continue
        denominator = float(comparator["scs"]) - float(row["receiver_scs"])
        if abs(denominator) < 1e-6:
            continue
        raw_effect = float(row["scs"]) - float(row["receiver_scs"])
        normalized.append(
            {
                **row,
                "raw_patch_scs_effect": raw_effect,
                "full_patch_scs_effect": denominator,
                "normalized_patch_effect": raw_effect / denominator,
            }
        )
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        grouped[(row["direction"], int(row["layer"]), row["position"])].append(row)
    cells = []
    for (direction, layer, position), values in sorted(grouped.items()):
        by_track: dict[str, list[float]] = defaultdict(list)
        for row in values:
            by_track[row["track_id"]].append(float(row["normalized_patch_effect"]))
        track_values = np.asarray([mean(items) for items in by_track.values()])
        estimate, low, high = _bootstrap(track_values, config)
        raw_by_track: dict[str, list[float]] = defaultdict(list)
        for row in values:
            raw_by_track[row["track_id"]].append(float(row["raw_patch_scs_effect"]))
        raw_values = np.asarray([mean(items) for items in raw_by_track.values()])
        raw_estimate, raw_low, raw_high = _bootstrap(raw_values, config)
        threshold = float(config["patching"].get("min_stable_full_patch_scs_effect", 0.10))
        stable_by_track: dict[str, list[float]] = defaultdict(list)
        for row in values:
            if abs(float(row["full_patch_scs_effect"])) >= threshold:
                stable_by_track[row["track_id"]].append(float(row["normalized_patch_effect"]))
        stable_values = np.asarray([mean(items) for items in stable_by_track.values()])
        stable_estimate, stable_low, stable_high = _bootstrap(stable_values, config)
        cells.append(
            {
                "direction": direction,
                "layer": layer,
                "position": position,
                "estimate": estimate,
                "median": float(np.median(track_values)) if len(track_values) else None,
                "ci95_low": low,
                "ci95_high": high,
                "stable_denominator_estimate": stable_estimate,
                "stable_denominator_ci95_low": stable_low,
                "stable_denominator_ci95_high": stable_high,
                "stable_denominator_threshold": threshold,
                "n_stable_denominator_tracks": len(stable_values),
                "raw_scs_effect": raw_estimate,
                "raw_scs_effect_ci95_low": raw_low,
                "raw_scs_effect_ci95_high": raw_high,
                "median_absolute_full_patch_scs_effect": float(
                    np.median([abs(float(row["full_patch_scs_effect"])) for row in values])
                ),
                "n_pairs": len(values),
                "n_tracks": len(track_values),
            }
        )
    self_rows = [row for row in rows if str(row["position"]).startswith("self_")]
    self_deviation = [abs(float(row["scs"]) - float(row["receiver_scs"])) for row in self_rows if row.get("scs") is not None and row.get("receiver_scs") is not None]
    summary = {
        "experiment": "e6",
        "n_rows": len(rows),
        "n_normalized_rows": len(normalized),
        "normalized_cells": cells,
        "self_patch_mean_absolute_scs_deviation": mean(self_deviation) if self_deviation else None,
        "normalization": "(window_patch_scs - receiver_scs) / (same_layer_full_real_patch_scs - receiver_scs)",
    }
    destination = root / "summary.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _bootstrap(values: np.ndarray, config: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    if not len(values):
        return None, None, None
    rng = np.random.default_rng(int(config["project"]["seed"]))
    count = int(config["scoring"]["bootstrap_replicates"])
    draws = rng.choice(values, size=(count, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E6 symmetric normalized encoder causal tracing")
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-pairs", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    path = prepare(config) if args.command == "prepare" else run(config, limit_pairs=args.limit_pairs) if args.command == "run" else analyze(config)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
