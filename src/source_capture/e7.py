from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import yaml
from scipy.optimize import minimize

from .records import read_jsonl, write_jsonl


MODEL_FEATURES = {
    "M0_local": ("current", "global", "current_sq", "global_sq", "current_x_global"),
    "M1_position": ("current", "onset", "rest", "onset_minus_rest", "onset_x_rest"),
    "M2_commitment": ("current", "onset_minus_rest", "history", "history_x_current", "lexical_degradation"),
}


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("E7 config must be a mapping")
    return value


def prepare(config: dict[str, Any], *, holdout: bool = False) -> Path:
    project = config["project"]
    root = Path(project["e1_e3_root"])
    split = "holdout" if holdout else "development"
    e4_root = Path(project["holdout_e4_root"] if holdout else project["e4_root"])
    e5_root = Path(project["holdout_e5_root"] if holdout else project["e5_root"])
    sources = {
        "e1": root / "e1" / "scores" / "whisper.jsonl",
        "e2": root / "e2" / "scores" / "whisper.jsonl",
        "e3": root / "e3" / "scores" / "whisper.jsonl",
        "e4a": e4_root / "e4a" / "runs" / "whisper.jsonl",
        "e4b": e4_root / "e4b" / "runs" / "whisper.jsonl",
        "e5": e5_root / "runs" / "whisper.jsonl",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("E7 inputs are incomplete:\n" + "\n".join(missing))
    e3_lookup = {
        row["sample_id"]: row
        for row in read_jsonl(sources["e3"])
        if row.get("split") == split
    }
    e5_solo = {
        (row["vocal_crop_id"], row["variant"]): row.get("lyric_recall")
        for row in read_jsonl(sources["e5"])
        if row.get("mode") == "solo" and row.get("score_error") is None
    }
    intact_recall = {
        crop: value
        for (crop, variant), value in e5_solo.items()
        if variant == "intact" and value is not None
    }
    records: list[dict[str, Any]] = []
    for family, path in sources.items():
        for row in read_jsonl(path):
            if row.get("split") != split or row.get("score_error") is not None:
                continue
            if family == "e5" and row.get("mode") != "mix":
                continue
            grounded = int(row.get("n_grounded") or 0)
            lyric = int(row.get("n_lyric") or 0)
            if grounded <= 0:
                continue
            enriched = e3_lookup.get(row.get("sample_id"), row) if family == "e4a" else row
            acoustic = _acoustic_features(enriched, family)
            history = _history_feature(row, family)
            lexical = 0.0
            if family == "e5":
                recall = e5_solo.get((row["vocal_crop_id"], row["variant"]))
                intact = intact_recall.get(row["vocal_crop_id"])
                if recall is not None and intact is not None:
                    lexical = float(recall) - float(intact)
            base = {
                "row_id": f"{family}:{row.get('sample_id', row.get('chunk_id'))}:{row.get('prefix_condition', row.get('history_condition', row.get('variant', '')))}",
                "family": family,
                "track_id": row["track_id"],
                "pair_id": row.get("pair_id"),
                "n_lyric": lyric,
                "n_grounded": grounded,
                "lyric_fraction": lyric / grounded,
                **acoustic,
                "history": history,
                "history_x_current": history * acoustic["current"],
                "lexical_degradation": lexical,
            }
            base.update(
                current_sq=base["current"] ** 2,
                global_sq=base["global"] ** 2,
                current_x_global=base["current"] * base["global"],
                onset_minus_rest=base["onset"] - base["rest"],
                onset_x_rest=base["onset"] * base["rest"],
            )
            records.append(base)
    destination = Path(project["output_root"]) / (
        "modeling_rows.holdout.jsonl" if holdout else "modeling_rows.jsonl"
    )
    write_jsonl(destination, records)
    return destination


def _acoustic_features(row: dict[str, Any], family: str) -> dict[str, float]:
    if family in {"e1", "e2"}:
        snr = float(row["snr_db"])
        attenuation = float(row.get("attenuation_db") or 0.0)
        intervals = row.get("intervention_intervals_s") or []
        segments = _segments_from_intervals(snr, attenuation, intervals)
    elif family in {"e3", "e4a"}:
        segments = [
            (float(item["start_s"]), float(item["end_s"]), float(item["snr_db"]))
            for item in row.get("snr_segments") or []
        ]
        if not segments:
            condition = row.get("condition")
            default = -10.0 if "lyric" in str(condition) else 10.0 if "speech" in str(condition) else 0.0
            segments = [(0.0, 10.0, default)]
    elif family == "e4b":
        local_snr = -10.0 if row["direction"] == "s_to_l" else 10.0
        segments = [(0.0, 10.0 - float(row["switch_s"]), local_snr)]
    elif family == "e5":
        segments = [(0.0, 10.0, float(row["snr_db"]))]
    else:
        raise ValueError(family)
    duration = max(end for _, end, _ in segments)
    onset_end = min(2.0, duration)
    current_start = max(0.0, duration - 2.0)
    return {
        "global": _window_lyric_advantage(segments, 0.0, duration),
        "onset": _window_lyric_advantage(segments, 0.0, onset_end),
        "rest": _window_lyric_advantage(segments, onset_end, duration),
        "current": _window_lyric_advantage(segments, current_start, duration),
    }


def _segments_from_intervals(snr: float, attenuation: float, intervals: list[list[float]]) -> list[tuple[float, float, float]]:
    boundaries = {0.0, 10.0}
    for start, end in intervals:
        boundaries.update((float(start), float(end)))
    ordered = sorted(boundaries)
    segments = []
    for start, end in zip(ordered[:-1], ordered[1:]):
        midpoint = (start + end) / 2
        treated = any(float(a) <= midpoint < float(b) for a, b in intervals)
        effective_snr = snr + attenuation if treated else snr
        segments.append((start, end, effective_snr))
    return segments


def _window_lyric_advantage(segments: list[tuple[float, float, float]], start: float, end: float) -> float:
    if end <= start:
        return 0.0
    power = 0.0
    covered = 0.0
    for seg_start, seg_end, snr in segments:
        overlap = max(0.0, min(end, seg_end) - max(start, seg_start))
        if overlap:
            power += overlap * 10.0 ** (-snr / 10.0)
            covered += overlap
    if covered <= 0:
        return 0.0
    return 10.0 * float(np.log10(max(power / covered, 1e-12)))


def _history_feature(row: dict[str, Any], family: str) -> float:
    if family == "e3":
        condition = str(row.get("condition", ""))
        switch = float(row.get("switch_s") or 0.0)
        magnitude = min(1.0, switch / 4.0) if switch else 0.0
        if condition in {"s_to_l", "symmetric_s_to_l"}:
            return -magnitude
        if condition in {"l_to_s", "symmetric_l_to_s"}:
            return magnitude
        return 0.0
    if family == "e4a":
        return {"speech": -1.0, "lyric": 1.0}.get(row.get("prefix_condition"), 0.0)
    if family == "e4b":
        condition = row.get("history_condition")
        if condition == "reset":
            return 0.0
        initial = -1.0 if row["direction"] == "s_to_l" else 1.0
        if condition == "carry_counterfactual":
            initial *= -1.0
        return initial * min(1.0, float(row["switch_s"]) / 4.0)
    return 0.0


class RidgeBinomial:
    def __init__(self, ridge: float, max_iterations: int):
        self.ridge = ridge
        self.max_iterations = max_iterations
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    def fit(self, x: np.ndarray, successes: np.ndarray, totals: np.ndarray) -> "RidgeBinomial":
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        standardized = (x - self.mean) / self.scale
        design = np.column_stack((np.ones(len(x)), standardized))
        y = successes / totals
        norm = float(totals.sum())

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            logits = design @ beta
            loss = float(np.sum(totals * (np.logaddexp(0.0, logits) - y * logits)) / norm)
            loss += 0.5 * self.ridge * float(np.dot(beta[1:], beta[1:]))
            probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))
            gradient = design.T @ (totals * (probability - y)) / norm
            gradient[1:] += self.ridge * beta[1:]
            return loss, gradient

        result = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": self.max_iterations})
        if not result.success:
            raise RuntimeError(f"model fit failed: {result.message}")
        self.coef = result.x
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.scale is not None and self.coef is not None
        design = np.column_stack((np.ones(len(x)), (x - self.mean) / self.scale))
        logits = design @ self.coef
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))


def fit(config: dict[str, Any]) -> Path:
    root = Path(config["project"]["output_root"])
    source = root / "modeling_rows.jsonl"
    if not source.exists():
        prepare(config)
    rows = list(read_jsonl(source))
    settings = config["fit"]
    folds = _track_folds(rows, int(settings["track_folds"]), int(config["project"]["seed"]))
    evaluations = []
    predictions = []
    for model_name, features in MODEL_FEATURES.items():
        for fold_name, train_indices, test_indices in folds + _family_folds(rows):
            model = RidgeBinomial(float(settings["ridge_lambda"]), int(settings["max_iterations"]))
            x_train, success_train, total_train = _arrays(rows, train_indices, features)
            x_test, success_test, total_test = _arrays(rows, test_indices, features)
            model.fit(x_train, success_train, total_train)
            probability = model.predict(x_test)
            metrics = _metrics(probability, success_test, total_test)
            evaluations.append({"model": model_name, "features": list(features), "n_parameters": len(features) + 1, "fold": fold_name, "n_train": len(train_indices), "n_test": len(test_indices), **metrics})
            for index, value in zip(test_indices, probability):
                predictions.append({"model": model_name, "fold": fold_name, "row_id": rows[index]["row_id"], "family": rows[index]["family"], "track_id": rows[index]["track_id"], "observed": rows[index]["lyric_fraction"], "predicted": float(value), "n_grounded": rows[index]["n_grounded"]})
    write_jsonl(root / "predictions.jsonl", predictions)
    final_models = {}
    all_indices = list(range(len(rows)))
    for model_name, features in MODEL_FEATURES.items():
        model = RidgeBinomial(
            float(settings["ridge_lambda"]), int(settings["max_iterations"])
        )
        x, successes, totals = _arrays(rows, all_indices, features)
        model.fit(x, successes, totals)
        assert model.mean is not None and model.scale is not None and model.coef is not None
        final_models[model_name] = {
            "features": list(features),
            "mean": model.mean.tolist(),
            "scale": model.scale.tolist(),
            "coef": model.coef.tolist(),
        }
    (root / "final_models.json").write_text(
        json.dumps(final_models, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = _summarize(evaluations, rows)
    destination = root / "summary.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def evaluate_holdout(config: dict[str, Any]) -> Path:
    root = Path(config["project"]["output_root"])
    source = root / "modeling_rows.holdout.jsonl"
    if not source.exists():
        prepare(config, holdout=True)
    model_path = root / "final_models.json"
    if not model_path.exists():
        fit(config)
    rows = list(read_jsonl(source))
    models = json.loads(model_path.read_text(encoding="utf-8"))
    predictions = []
    evaluations = []
    for model_name, state in models.items():
        features = tuple(state["features"])
        indices = list(range(len(rows)))
        x, successes, totals = _arrays(rows, indices, features)
        center = np.asarray(state["mean"], dtype=np.float64)
        scale = np.asarray(state["scale"], dtype=np.float64)
        coef = np.asarray(state["coef"], dtype=np.float64)
        design = np.column_stack((np.ones(len(x)), (x - center) / scale))
        probability = 1.0 / (1.0 + np.exp(-np.clip(design @ coef, -35, 35)))
        evaluations.append(
            {
                "model": model_name,
                "n_parameters": len(features) + 1,
                "n_test": len(rows),
                **_metrics(probability, successes, totals),
            }
        )
        for row, value in zip(rows, probability):
            predictions.append(
                {
                    "model": model_name,
                    "row_id": row["row_id"],
                    "family": row["family"],
                    "track_id": row["track_id"],
                    "observed": row["lyric_fraction"],
                    "predicted": float(value),
                    "n_grounded": row["n_grounded"],
                }
            )
    write_jsonl(root / "predictions.holdout.jsonl", predictions)
    summary = {
        "experiment": "e7_final_holdout",
        "n_rows": len(rows),
        "n_tracks": len({row["track_id"] for row in rows}),
        "families": {
            family: sum(row["family"] == family for row in rows)
            for family in sorted({row["family"] for row in rows})
        },
        "evaluations": evaluations,
    }
    destination = root / "holdout_summary.json"
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def _arrays(rows: list[dict[str, Any]], indices: list[int], features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([[float(rows[index][name]) for name in features] for index in indices], dtype=np.float64)
    successes = np.asarray([rows[index]["n_lyric"] for index in indices], dtype=np.float64)
    totals = np.asarray([rows[index]["n_grounded"] for index in indices], dtype=np.float64)
    return x, successes, totals


def _track_folds(rows: list[dict[str, Any]], count: int, seed: int) -> list[tuple[str, list[int], list[int]]]:
    tracks = sorted({row["track_id"] for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(tracks)
    assignments = {track: index % count for index, track in enumerate(tracks)}
    return [(f"track_{fold}", [i for i, row in enumerate(rows) if assignments[row["track_id"]] != fold], [i for i, row in enumerate(rows) if assignments[row["track_id"]] == fold]) for fold in range(count)]


def _family_folds(rows: list[dict[str, Any]]) -> list[tuple[str, list[int], list[int]]]:
    families = sorted({row["family"] for row in rows})
    return [(f"intervention_{family}", [i for i, row in enumerate(rows) if row["family"] != family], [i for i, row in enumerate(rows) if row["family"] == family]) for family in families]


def _metrics(probability: np.ndarray, successes: np.ndarray, totals: np.ndarray) -> dict[str, float]:
    observed = successes / totals
    clipped = np.clip(probability, 1e-8, 1 - 1e-8)
    log_loss = -float(np.sum(successes * np.log(clipped) + (totals - successes) * np.log(1 - clipped)) / totals.sum())
    brier = float(np.sum(totals * (probability - observed) ** 2) / totals.sum())
    truth = observed >= 0.5
    predicted = probability >= 0.5
    accuracy = float(np.mean(truth == predicted))
    f1_values = []
    for label in (False, True):
        tp = np.sum((truth == label) & (predicted == label))
        fp = np.sum((truth != label) & (predicted == label))
        fn = np.sum((truth == label) & (predicted != label))
        denominator = 2 * tp + fp + fn
        f1_values.append(float(2 * tp / denominator) if denominator else 0.0)
    ece = 0.0
    for lower in np.linspace(0, 0.9, 10):
        mask = (probability >= lower) & (probability < lower + 0.1 if lower < 0.9 else probability <= 1.0)
        if np.any(mask):
            weight = float(totals[mask].sum() / totals.sum())
            ece += weight * abs(float(np.average(observed[mask], weights=totals[mask])) - float(np.average(probability[mask], weights=totals[mask])))
    return {"heldout_token_nll": log_loss, "brier": brier, "accuracy": accuracy, "macro_f1": mean(f1_values), "ece10": ece}


def _summarize(evaluations: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    track = [row for row in evaluations if row["fold"].startswith("track_")]
    intervention = [row for row in evaluations if row["fold"].startswith("intervention_")]
    aggregates = []
    for kind, values in (("leave_tracks_out", track), ("leave_intervention_out", intervention)):
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            by_model[row["model"]].append(row)
        for model, model_rows in by_model.items():
            aggregates.append({"evaluation": kind, "model": model, "n_folds": len(model_rows), "n_parameters": model_rows[0]["n_parameters"], **{f"mean_{metric}": mean(row[metric] for row in model_rows) for metric in ("heldout_token_nll", "brier", "accuracy", "macro_f1", "ece10")}})
    return {"experiment": "e7", "n_rows": len(rows), "n_tracks": len({row["track_id"] for row in rows}), "families": {family: sum(row["family"] == family for row in rows) for family in sorted({row["family"] for row in rows})}, "models": {name: {"features": list(features), "n_parameters": len(features) + 1} for name, features in MODEL_FEATURES.items()}, "fold_results": evaluations, "aggregates": aggregates, "note": "M0/M1/M2 use the same binomial observation model, ridge penalty, and six-parameter budget including intercept."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E7 matched-budget mechanism model comparison")
    parser.add_argument(
        "command",
        choices=("prepare", "fit", "all", "prepare-holdout", "evaluate-holdout"),
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command in {"prepare", "all"}:
        print(prepare(config))
    if args.command in {"fit", "all"}:
        print(fit(config))
    if args.command == "prepare-holdout":
        print(prepare(config, holdout=True))
    if args.command == "evaluate-holdout":
        print(evaluate_holdout(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
