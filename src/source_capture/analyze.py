from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .config import experiment_dir
from .records import read_jsonl
from .scoring import score_time_window


METRICS = ("lir", "tsr", "scs", "no_grounded_output")


def analyze_experiment(config: dict[str, Any], experiment: str) -> tuple[Path, Path]:
    directory = experiment_dir(config, experiment)
    score_files = sorted((directory / "scores").glob("*.jsonl"))
    if not score_files:
        raise FileNotFoundError(f"no score files below {directory / 'scores'}")
    rows = [row for path in score_files for row in read_jsonl(path)]
    summary = {
        "experiment": experiment,
        "models": sorted({row["model"] for row in rows}),
        "n_rows": len(rows),
        "aggregates": _aggregates(rows, experiment),
    }
    if experiment == "e1":
        summary["primary_contrasts"] = _e1_contrasts(rows, config)
    elif experiment == "e2":
        summary["dose_analysis"] = _e2_analysis(rows)
    elif experiment == "e3":
        summary["history_effects"] = _e3_history_effects(rows, config)
        summary["symmetric_order_effects"] = _e3_symmetric_order_effects(
            rows, config
        )
    json_path = directory / "summary.json"
    csv_path = directory / "summary.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(csv_path, summary["aggregates"])
    return json_path, csv_path


def _aggregates(rows: list[dict[str, Any]], experiment: str) -> list[dict[str, Any]]:
    fields = {
        "e1": ["model", "split", "snr_db", "condition"],
        "e2": ["model", "split", "snr_db", "duration_s", "attenuation_db"],
        "e3": ["model", "split", "condition", "switch_s"],
    }[experiment]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        aggregate = {field: value for field, value in zip(fields, key)}
        aggregate["n"] = len(values)
        aggregate["n_tracks"] = len({row["track_id"] for row in values})
        for metric in METRICS:
            available = [float(row[metric]) for row in values if row.get(metric) is not None]
            aggregate[f"mean_{metric}"] = mean(available) if available else None
        if experiment == "e2":
            aggregate["mean_energy_dose"] = mean(float(row["energy_dose"]) for row in values)
            aggregate["mean_energy_dose_fraction"] = mean(
                float(row["energy_dose_fraction"]) for row in values
            )
        output.append(aggregate)
    return output


def _e1_contrasts(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    comparators = {"middle", "offset", "distributed"}
    grouped: dict[tuple[str, str, float], dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        grouped[(row["model"], row["split"], float(row["snr_db"]))][row["pair_id"]][
            row["condition"]
        ] = row
    output = []
    seed = int(config["project"]["seed"])
    replicates = int(config["scoring"].get("bootstrap_replicates", 10000))
    for (model, split, snr_db), pairs in sorted(grouped.items()):
        for metric in ("lir", "tsr"):
            track_effects: dict[str, list[float]] = defaultdict(list)
            for conditions in pairs.values():
                if "onset" not in conditions or not comparators <= set(conditions):
                    continue
                onset = conditions["onset"].get(metric)
                values = [conditions[name].get(metric) for name in comparators]
                if onset is None or any(value is None for value in values):
                    continue
                effect = float(onset) - mean(float(value) for value in values)
                track_effects[conditions["onset"]["track_id"]].append(effect)
            by_track = [mean(values) for values in track_effects.values()]
            estimate, low, high = _bootstrap_track_means(by_track, replicates, seed)
            output.append(
                {
                    "model": model,
                    "split": split,
                    "snr_db": snr_db,
                    "metric": metric,
                    "contrast": (
                        f"{metric.upper()}(onset)-"
                        f"mean({metric.upper()}(middle,offset,distributed))"
                    ),
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_tracks": len(by_track),
                }
            )
    return output


def _e2_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("lir") is not None:
            grouped[(row["model"], row["split"], float(row["snr_db"]))].append(row)
    output = []
    for (model, split, snr), values in sorted(grouped.items()):
        dose = np.array([float(row["energy_dose_fraction"]) for row in values])
        lir = np.array([float(row["lir"]) for row in values])
        tsr_rows = [row for row in values if row.get("tsr") is not None]
        dose_tsr = np.array([float(row["energy_dose_fraction"]) for row in tsr_rows])
        tsr = np.array([float(row["tsr"]) for row in tsr_rows])
        output.append(
            {
                "model": model,
                "split": split,
                "snr_db": snr,
                "n": len(values),
                "spearman_dose_lir": _spearman(dose, lir),
                "spearman_dose_tsr": _spearman(dose_tsr, tsr),
                "note": "Descriptive only; fit linear/Hill/change-point models after development data are complete.",
            }
        )
    return output


def _e3_history_effects(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    by_key = {
        (row["model"], row["split"], row["pair_id"], row["condition_id"]): row
        for row in rows
    }
    pair_effects = []
    missing_alignment: list[str] = []
    for row in rows:
        comparator_id = row.get("static_comparator_condition")
        if row.get("condition") not in {"s_to_l", "l_to_s"} or not comparator_id:
            continue
        comparator = by_key.get((row["model"], row["split"], row["pair_id"], comparator_id))
        if comparator is None:
            continue
        if (
            not row.get("words")
            or not comparator.get("words")
            or not row.get("speech_words")
            or not row.get("lyric_words")
        ):
            missing_alignment.append(f"{row['model']}:{row['sample_id']}")
            continue
        start, end = float(row["evaluation_start_s"]), float(row["evaluation_end_s"])
        switched_window = score_time_window(
            row.get("words"),
            start,
            end,
            row["speech_reference"],
            row["lyric_reference"],
            config["scoring"],
            speech_words=row.get("speech_words"),
            lyric_words=row.get("lyric_words"),
        )
        comparator_window = score_time_window(
            comparator.get("words"),
            start,
            end,
            comparator["speech_reference"],
            comparator["lyric_reference"],
            config["scoring"],
            speech_words=comparator.get("speech_words"),
            lyric_words=comparator.get("lyric_words"),
        )
        if switched_window is None or comparator_window is None:
            continue
        if switched_window["scs"] is None or comparator_window["scs"] is None:
            continue
        pair_effects.append(
            {
                "model": row["model"],
                "split": row["split"],
                "pair_id": row["pair_id"],
                "track_id": row["track_id"],
                "condition": row["condition"],
                "switch_s": row["switch_s"],
                "evaluation_start_s": start,
                "evaluation_end_s": end,
                "history_effect": float(switched_window["scs"])
                - float(comparator_window["scs"]),
                "expected_direction": row["history_effect_direction"],
            }
        )
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_effects:
        grouped[(row["model"], row["split"], row["condition"], float(row["switch_s"]))].append(row)
    aggregate = []
    seed = int(config["project"]["seed"])
    replicates = int(config["scoring"].get("bootstrap_replicates", 10000))
    for (model, split, condition, switch_s), values in sorted(grouped.items()):
        tracks: dict[str, list[float]] = defaultdict(list)
        for value in values:
            tracks[value["track_id"]].append(float(value["history_effect"]))
        by_track = [mean(track_values) for track_values in tracks.values()]
        estimate, low, high = _bootstrap_track_means(by_track, replicates, seed)
        aggregate.append(
            {
                "model": model,
                "split": split,
                "condition": condition,
                "switch_s": switch_s,
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "n_tracks": len(by_track),
                "expected_direction": values[0]["expected_direction"],
            }
        )
    return {
        "aggregate": aggregate,
        "pair_effects": pair_effects,
        "missing_alignment_rows": len(missing_alignment),
        "missing_alignment_examples": missing_alignment[:10],
        "note": (
            "Rows without model output timestamps or aligned speech/lyric references "
            "are excluded from window-level history effects; clip-level scores remain valid."
        ),
    }


def _e3_symmetric_order_effects(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("condition_id") in {"s5_to_l5", "l5_to_s5"}:
            grouped[(row["model"], row["split"], row["pair_id"])][
                row["condition_id"]
            ] = row
    by_group: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (model, split, _), conditions in grouped.items():
        if not {"s5_to_l5", "l5_to_s5"} <= set(conditions):
            continue
        left = conditions["s5_to_l5"].get("scs")
        right = conditions["l5_to_s5"].get("scs")
        if left is None or right is None:
            continue
        track = conditions["s5_to_l5"]["track_id"]
        by_group[(model, split)][track].append(float(left) - float(right))
    seed = int(config["project"]["seed"])
    replicates = int(config["scoring"].get("bootstrap_replicates", 10000))
    output = []
    for (model, split), tracks in sorted(by_group.items()):
        values = [mean(effects) for effects in tracks.values()]
        estimate, low, high = _bootstrap_track_means(values, replicates, seed)
        output.append(
            {
                "model": model,
                "split": split,
                "contrast": "SCS(S5->L5)-SCS(L5->S5)",
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "n_tracks": len(values),
            }
        )
    return output


def _bootstrap_track_means(
    values: list[float], replicates: int, seed: int
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(replicates, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    from scipy.stats import spearmanr

    value = spearmanr(x, y).statistic
    return None if not np.isfinite(value) else float(value)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
