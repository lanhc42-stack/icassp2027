from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import yaml

from .audio import read_pcm16
from .records import append_jsonl, read_jsonl, stable_id, write_jsonl
from .scoring import attribute_tokens, normalize_words


PREFIX_CONDITIONS = ("none", "speech", "lyric", "unrelated")


def load_e4_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("E4 config must be a mapping")
    for key in ("project", "model", "scoring", "e4a", "e4b"):
        if key not in config:
            raise ValueError(f"E4 config is missing {key}")
    return config


def _e3_scores(config: dict[str, Any]) -> Path:
    return Path(config["project"]["e3_root"]) / "scores" / "whisper.jsonl"


def _output_root(config: dict[str, Any]) -> Path:
    return Path(config["project"]["output_root"])


def _eligible_development_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = _e3_scores(config)
    if not source.exists():
        raise FileNotFoundError(source)
    return [
        row
        for row in read_jsonl(source)
        if row.get("split") == config["project"].get("split", "development")
        and row.get("error") is None
        and row.get("score_error") is None
        and row.get("scs") is not None
    ]


def prepare_e4a(config: dict[str, Any]) -> Path:
    settings = config["e4a"]
    candidates = [
        row
        for row in _eligible_development_rows(config)
        if row.get("condition") == settings["source_condition"]
    ]
    if not candidates:
        raise RuntimeError("no eligible E4a source rows")
    count = int(settings["samples_per_stratum"])
    cap = int(settings["max_samples_per_track_per_stratum"])
    groups = {
        "boundary": sorted(
            candidates,
            key=lambda row: (abs(float(row["scs"])), row["sample_id"]),
        ),
        "stable_speech": sorted(
            (
                row
                for row in candidates
                if float(row["scs"])
                <= float(settings["strata"]["stable_speech"]["scs_max"])
            ),
            key=lambda row: (float(row["scs"]), row["sample_id"]),
        ),
        "stable_lyric": sorted(
            (
                row
                for row in candidates
                if float(row["scs"])
                >= float(settings["strata"]["stable_lyric"]["scs_min"])
            ),
            key=lambda row: (-float(row["scs"]), row["sample_id"]),
        ),
    }
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for stratum, rows in groups.items():
        per_track: dict[str, int] = defaultdict(int)
        chosen: list[dict[str, Any]] = []
        for row in rows:
            if row["sample_id"] in used_ids:
                continue
            track = str(row["track_id"])
            if per_track[track] >= cap:
                continue
            chosen.append(row)
            per_track[track] += 1
            used_ids.add(row["sample_id"])
            if len(chosen) == count:
                break
        if len(chosen) < count:
            raise RuntimeError(
                f"E4a stratum {stratum} only yielded {len(chosen)}/{count} rows"
            )
        selected.extend({**row, "selection_stratum": stratum} for row in chosen)

    speech_pool = [str(row["speech_reference"]) for row in candidates]
    rng = random.Random(int(config["project"]["seed"]))
    prefix_words = int(settings["prefix_words"])
    manifest: list[dict[str, Any]] = []
    for base in selected:
        speech_prefix = _first_words(base["speech_reference"], prefix_words)
        lyric_prefix = _first_words(base["lyric_reference"], prefix_words)
        unrelated_options = [
            text for text in speech_pool if text != base["speech_reference"]
        ]
        unrelated_prefix = _first_words(rng.choice(unrelated_options), prefix_words)
        prefixes = {
            "none": "",
            "speech": speech_prefix,
            "lyric": lyric_prefix,
            "unrelated": unrelated_prefix,
        }
        for condition in PREFIX_CONDITIONS:
            manifest.append(
                {
                    **_base_fields(base),
                    "experiment": "e4a",
                    "selection_stratum": base["selection_stratum"],
                    "selection_free_decode_scs": base["scs"],
                    "prefix_condition": condition,
                    "prefix_text": prefixes[condition],
                    "intervention_id": stable_id(
                        "e4a", base["sample_id"], condition
                    ),
                }
            )
    destination = _output_root(config) / "e4a" / "manifest.jsonl"
    write_jsonl(destination, manifest)
    return destination


def prepare_e4b(config: dict[str, Any]) -> Path:
    settings = config["e4b"]
    wanted_switches = {float(value) for value in settings["switch_s"]}
    wanted_directions = set(settings["directions"])
    candidates = [
        row
        for row in _eligible_development_rows(config)
        if row.get("condition") in wanted_directions
        and float(row.get("switch_s", -1)) in wanted_switches
    ]
    by_track_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_track_pair[(str(row["track_id"]), str(row["pair_id"]))].append(row)
    required = {(direction, switch) for direction in wanted_directions for switch in wanted_switches}
    complete: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for (track, pair), rows in by_track_pair.items():
        available = {(row["condition"], float(row["switch_s"])) for row in rows}
        if required <= available:
            complete[track].append((pair, rows))
    pairs_per_track = int(settings["pairs_per_track"])
    selected: list[dict[str, Any]] = []
    for track in sorted(complete):
        ordered = sorted(
            complete[track], key=lambda item: stable_id("e4b", track, item[0])
        )
        for _, rows in ordered[:pairs_per_track]:
            selected.extend(rows)
    if not selected:
        raise RuntimeError("no complete E4b track/pair blocks")
    manifest = [
        {
            **_base_fields(row),
            "experiment": "e4b",
            "direction": row["condition"],
            "switch_s": float(row["switch_s"]),
            "chunk_id": stable_id(
                "e4b", row["sample_id"], row["condition"], row["switch_s"]
            ),
        }
        for row in sorted(
            selected,
            key=lambda row: (
                row["track_id"],
                row["pair_id"],
                row["condition"],
                float(row["switch_s"]),
            ),
        )
    ]
    destination = _output_root(config) / "e4b" / "manifest.jsonl"
    write_jsonl(destination, manifest)
    return destination


def _base_fields(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "sample_id",
        "pair_id",
        "family_id",
        "track_id",
        "speech_id",
        "split",
        "audio_path",
        "audio_sha256",
        "speech_reference",
        "lyric_reference",
        "speech_words",
        "lyric_words",
        "condition",
        "condition_id",
    )
    return {key: row.get(key) for key in fields}


def _first_words(text: str, count: int) -> str:
    return " ".join(str(text).split()[:count])


class WhisperPromptRunner:
    def __init__(self, config: dict[str, Any]):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.config = config
        self.device = str(config.get("device", "cuda"))
        self.dtype = getattr(torch, str(config.get("dtype", "float16")))
        self.processor = WhisperProcessor.from_pretrained(config["model_path"])
        self.model = WhisperForConditionalGeneration.from_pretrained(
            config["model_path"], torch_dtype=self.dtype
        ).to(self.device).eval()

    def transcribe(self, audio: np.ndarray, sample_rate: int, prompt: str) -> dict[str, Any]:
        torch = self.torch
        encoded = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        features = encoded.input_features.to(device=self.device, dtype=self.dtype)
        prompt_ids = None
        prompt_length = 0
        prompt_logprob = None
        if prompt.strip():
            prompt_ids = self.processor.get_prompt_ids(
                prompt, return_tensors="pt"
            ).to(self.device)
            prompt_length = int(prompt_ids.numel())
            prompt_logprob = self._prompt_logprob(features, prompt_ids)
        kwargs: dict[str, Any] = {
            "language": self.config.get("language", "en"),
            "task": self.config.get("task", "transcribe"),
            "max_new_tokens": int(self.config.get("max_new_tokens", 180)),
            "return_timestamps": False,
        }
        if prompt_ids is not None:
            kwargs["prompt_ids"] = prompt_ids
        with torch.inference_mode():
            sequence = self.model.generate(features, **kwargs)[0]
        free_tokens = sequence[prompt_length:]
        hypothesis = self.processor.tokenizer.decode(
            free_tokens, skip_special_tokens=True
        ).strip()
        return {
            "hyp": hypothesis,
            "prompt_token_count_including_startofprev": prompt_length,
            "mean_prompt_token_logprob": prompt_logprob,
        }

    def _prompt_logprob(self, features: Any, prompt_ids: Any) -> float | None:
        if int(prompt_ids.numel()) < 2:
            return None
        torch = self.torch
        decoder_input_ids = prompt_ids[:-1].unsqueeze(0)
        targets = prompt_ids[1:]
        with torch.inference_mode():
            logits = self.model(
                input_features=features,
                decoder_input_ids=decoder_input_ids,
            ).logits[0]
            token_logprobs = logits.float().log_softmax(dim=-1).gather(
                1, targets.unsqueeze(1)
            )
        return float(token_logprobs.mean().item())


def run_e4a(config: dict[str, Any], *, limit: int | None = None) -> Path:
    directory = _output_root(config) / "e4a"
    manifest = directory / "manifest.jsonl"
    if not manifest.exists():
        prepare_e4a(config)
    output = directory / "runs" / "whisper.jsonl"
    completed = {
        row["intervention_id"] for row in read_jsonl(output)
    } if output.exists() else set()
    rows = [
        row for row in read_jsonl(manifest)
        if row["intervention_id"] not in completed
    ]
    if limit is not None:
        rows = rows[:limit]
    runner = WhisperPromptRunner(config["model"])
    start = time.time()
    for index, row in enumerate(rows, 1):
        audio, sample_rate = read_pcm16(row["audio_path"])
        result = _safe_transcribe(runner, audio, sample_rate, row["prefix_text"])
        score = _score_result(row, result, config["scoring"])
        append_jsonl(output, {**row, **result, **score, "model": "whisper"})
        _progress(index, len(rows), start)
    return output


def run_e4b(config: dict[str, Any], *, limit: int | None = None) -> Path:
    directory = _output_root(config) / "e4b"
    manifest = directory / "manifest.jsonl"
    if not manifest.exists():
        prepare_e4b(config)
    output = directory / "runs" / "whisper.jsonl"
    completed = {
        (row["chunk_id"], row["history_condition"])
        for row in read_jsonl(output)
    } if output.exists() else set()
    rows = list(read_jsonl(manifest))
    runner = WhisperPromptRunner(config["model"])
    start = time.time()
    written = 0
    total = sum(
        1 for row in rows for condition in ("reset", "carry_natural", "carry_counterfactual")
        if (row["chunk_id"], condition) not in completed
    )
    if limit is not None:
        total = min(total, limit)
    for row in rows:
        if limit is not None and written >= limit:
            break
        audio, sample_rate = read_pcm16(row["audio_path"])
        boundary = round(float(row["switch_s"]) * sample_rate)
        chunk_a, chunk_b = audio[:boundary], audio[boundary:]
        natural = _safe_transcribe(runner, chunk_a, sample_rate, "")
        natural_prompt = _last_words(
            natural.get("hyp", ""), int(config["e4b"]["natural_prompt_max_words"])
        )
        counterfactual_prompt = _counterfactual_prompt(
            row,
            natural_prompt,
            config["scoring"],
        )
        prompts = {
            "reset": "",
            "carry_natural": natural_prompt,
            "carry_counterfactual": counterfactual_prompt,
        }
        speech_b = _window_reference(row.get("speech_words"), float(row["switch_s"]), 10.0)
        lyric_b = _window_reference(row.get("lyric_words"), float(row["switch_s"]), 10.0)
        scoring_row = {
            **row,
            "speech_reference": speech_b or row["speech_reference"],
            "lyric_reference": lyric_b or row["lyric_reference"],
        }
        for condition, prompt in prompts.items():
            if limit is not None and written >= limit:
                break
            key = (row["chunk_id"], condition)
            if key in completed:
                continue
            result = _safe_transcribe(runner, chunk_b, sample_rate, prompt)
            score = _score_result(scoring_row, result, config["scoring"])
            append_jsonl(
                output,
                {
                    **scoring_row,
                    **result,
                    **score,
                    "model": "whisper",
                    "history_condition": condition,
                    "history_text": prompt,
                    "natural_a_hyp": natural.get("hyp", ""),
                    "natural_a_error": natural.get("error"),
                },
            )
            written += 1
            _progress(written, total, start)
    return output


def _safe_transcribe(
    runner: WhisperPromptRunner,
    audio: np.ndarray,
    sample_rate: int,
    prompt: str,
) -> dict[str, Any]:
    started = time.time()
    try:
        result = runner.transcribe(audio, sample_rate, prompt)
        return {**result, "error": None, "inference_seconds": time.time() - started}
    except Exception as exc:
        return {
            "hyp": "",
            "error": f"{type(exc).__name__}: {exc}",
            "inference_seconds": time.time() - started,
            "prompt_token_count_including_startofprev": None,
            "mean_prompt_token_logprob": None,
        }


def _score_result(
    row: dict[str, Any], result: dict[str, Any], scoring: dict[str, Any]
) -> dict[str, Any]:
    if result.get("error"):
        return {
            "lir": None,
            "tsr": None,
            "scs": None,
            "n_grounded": 0,
            "score_error": "inference error",
        }
    score = attribute_tokens(
        result.get("hyp", ""),
        row.get("speech_reference", ""),
        row.get("lyric_reference", ""),
        scoring,
    )
    score["score_error"] = None
    return score


def _last_words(text: str, count: int) -> str:
    words = str(text).split()
    return " ".join(words[-count:])


def _counterfactual_prompt(
    row: dict[str, Any], natural_prompt: str, scoring: dict[str, Any]
) -> str:
    count = max(3, min(30, len(normalize_words(natural_prompt, scoring))))
    source = "lyric_words" if row["direction"] == "s_to_l" else "speech_words"
    words = [
        str(item.get("word", ""))
        for item in row.get(source) or []
        if float(item.get("start", 1e9)) < float(row["switch_s"])
    ]
    if not words:
        reference = (
            row["lyric_reference"]
            if row["direction"] == "s_to_l"
            else row["speech_reference"]
        )
        words = str(reference).split()
    return " ".join(words[-count:])


def _window_reference(
    words: list[dict[str, Any]] | None, start_s: float, end_s: float
) -> str:
    if not words:
        return ""
    return " ".join(
        str(item.get("word", ""))
        for item in words
        if float(item.get("end", -1)) > start_s
        and float(item.get("start", 1e9)) < end_s
    )


def _progress(index: int, total: int, started: float) -> None:
    if index % 10 == 0 or index == total:
        print(f"[{index}/{total}] {time.time() - started:.1f}s", flush=True)


def analyze_e4(config: dict[str, Any], experiment: str) -> Path:
    source = _output_root(config) / experiment / "runs" / "whisper.jsonl"
    if not source.exists():
        raise FileNotFoundError(source)
    rows = [row for row in read_jsonl(source) if row.get("score_error") is None]
    if experiment == "e4a":
        summary = _analyze_e4a(rows, config)
    else:
        summary = _analyze_e4b(rows, config)
    destination = _output_root(config) / experiment / "summary.json"
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def _analyze_e4a(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    grouped = _group(rows, ("selection_stratum", "prefix_condition"))
    aggregates = [_aggregate(key, values, ("selection_stratum", "prefix_condition")) for key, values in grouped.items()]
    by_sample = _group(rows, ("sample_id",))
    contrasts_available = []
    contrasts_zero_filled = []
    for stratum in ("boundary", "stable_speech", "stable_lyric"):
        relevant = [values for values in by_sample.values() if values[0]["selection_stratum"] == stratum]
        for left, right in (("lyric", "speech"), ("speech", "none"), ("lyric", "none"), ("unrelated", "none")):
            effects: list[tuple[str, float]] = []
            for values in relevant:
                conditions = {row["prefix_condition"]: row for row in values}
                if left in conditions and right in conditions:
                    a, b = conditions[left].get("scs"), conditions[right].get("scs")
                    if a is not None and b is not None:
                        effects.append((conditions[left]["track_id"], float(a) - float(b)))
            contrasts_available.append(_contrast_record(stratum, left, right, effects, config))
            zero_filled = []
            for values in relevant:
                conditions = {row["prefix_condition"]: row for row in values}
                if left in conditions and right in conditions:
                    a = float(conditions[left]["scs"]) if conditions[left].get("scs") is not None else 0.0
                    b = float(conditions[right]["scs"]) if conditions[right].get("scs") is not None else 0.0
                    zero_filled.append((conditions[left]["track_id"], a - b))
            contrasts_zero_filled.append(_contrast_record(stratum, left, right, zero_filled, config))
    return {
        "experiment": "e4a",
        "n_rows": len(rows),
        "aggregates": aggregates,
        "contrasts_available_case": contrasts_available,
        "contrasts_zero_filled": contrasts_zero_filled,
        "missing_output_policy": "Available-case SCS and SCS=0 for no-grounded-output are both reported; no-grounded-output rate is explicit in aggregates.",
    }


def _analyze_e4b(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    grouped = _group(rows, ("direction", "switch_s", "history_condition"))
    aggregates = [_aggregate(key, values, ("direction", "switch_s", "history_condition")) for key, values in grouped.items()]
    by_chunk = _group(rows, ("chunk_id",))
    contrasts_available = []
    contrasts_zero_filled = []
    for direction in config["e4b"]["directions"]:
        for switch_s in config["e4b"]["switch_s"]:
            effects_by_name: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for values in by_chunk.values():
                first = values[0]
                if first["direction"] != direction or float(first["switch_s"]) != float(switch_s):
                    continue
                conditions = {row["history_condition"]: row for row in values}
                for left, right in (("carry_natural", "reset"), ("carry_counterfactual", "carry_natural")):
                    if left in conditions and right in conditions:
                        a, b = conditions[left].get("scs"), conditions[right].get("scs")
                        if a is not None and b is not None:
                            effects_by_name[f"{left}_minus_{right}"].append((first["track_id"], float(a) - float(b)))
            for name, effects in effects_by_name.items():
                left, right = name.split("_minus_")
                record = _contrast_record(direction, left, right, effects, config)
                record["switch_s"] = float(switch_s)
                contrasts_available.append(record)
            for left, right in (("carry_natural", "reset"), ("carry_counterfactual", "carry_natural")):
                effects = []
                for values in by_chunk.values():
                    first = values[0]
                    if first["direction"] != direction or float(first["switch_s"]) != float(switch_s):
                        continue
                    conditions = {row["history_condition"]: row for row in values}
                    if left in conditions and right in conditions:
                        a = float(conditions[left]["scs"]) if conditions[left].get("scs") is not None else 0.0
                        b = float(conditions[right]["scs"]) if conditions[right].get("scs") is not None else 0.0
                        effects.append((first["track_id"], a - b))
                record = _contrast_record(direction, left, right, effects, config)
                record["switch_s"] = float(switch_s)
                contrasts_zero_filled.append(record)
    return {
        "experiment": "e4b",
        "n_rows": len(rows),
        "aggregates": aggregates,
        "contrasts_available_case": contrasts_available,
        "contrasts_zero_filled": contrasts_zero_filled,
        "missing_output_policy": "Available-case SCS and SCS=0 for no-grounded-output are both reported; no-grounded-output rate is explicit in aggregates.",
    }


def _group(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    output: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[tuple(row.get(field) for field in fields)].append(row)
    return output


def _aggregate(key: tuple[Any, ...], rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    record = dict(zip(fields, key))
    record["n"] = len(rows)
    record["n_tracks"] = len({row["track_id"] for row in rows})
    for metric in ("lir", "tsr", "scs"):
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        record[f"mean_{metric}"] = mean(values) if values else None
    record["mean_no_grounded_output"] = mean(
        float(bool(row.get("no_grounded_output"))) for row in rows
    )
    prompt_logprobs = [
        float(row["mean_prompt_token_logprob"])
        for row in rows
        if row.get("mean_prompt_token_logprob") is not None
    ]
    record["mean_prompt_token_logprob"] = (
        mean(prompt_logprobs) if prompt_logprobs else None
    )
    return record


def _contrast_record(
    stratum: str,
    left: str,
    right: str,
    effects: list[tuple[str, float]],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_track: dict[str, list[float]] = defaultdict(list)
    for track, effect in effects:
        by_track[track].append(effect)
    track_means = [mean(values) for values in by_track.values()]
    estimate, low, high = _bootstrap(
        track_means,
        int(config["scoring"].get("bootstrap_replicates", 10000)),
        int(config["project"]["seed"]),
    )
    return {
        "stratum_or_direction": stratum,
        "contrast": f"SCS({left})-SCS({right})",
        "estimate": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "n_pairs": len(effects),
        "n_tracks": len(track_means),
    }


def _bootstrap(values: list[float], replicates: int, seed: int) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(replicates, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E4 decoder-history causal interventions")
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", choices=("e4a", "e4b", "all"), required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_e4_config(args.config)
    experiments = ("e4a", "e4b") if args.experiment == "all" else (args.experiment,)
    for experiment in experiments:
        if args.command == "prepare":
            path = prepare_e4a(config) if experiment == "e4a" else prepare_e4b(config)
        elif args.command == "run":
            path = run_e4a(config, limit=args.limit) if experiment == "e4a" else run_e4b(config, limit=args.limit)
        else:
            path = analyze_e4(config, experiment)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
