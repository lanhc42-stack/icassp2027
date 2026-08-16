from __future__ import annotations

import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from .config import experiment_dir
from .records import load_json_or_jsonl, read_jsonl, write_jsonl


def normalize_words(text: str, config: dict[str, Any]) -> list[str]:
    value = text or ""
    if config.get("lowercase", True):
        value = value.lower()
    if config.get("strip_punctuation", True):
        value = value.translate(str.maketrans("", "", string.punctuation))
    return [word for word in re.split(r"\s+", value.strip()) if word]


def attribute_tokens(
    hypothesis: str,
    speech_reference: str,
    lyric_reference: str,
    scoring_config: dict[str, Any],
) -> dict[str, Any]:
    """Conservative dual-reference attribution using exclusive token capacity.

    Common words present in both references are marked ambiguous and excluded
    from LIR/SCS. Repeated matches cannot exceed their reference multiplicity.
    """
    hypothesis_words = normalize_words(hypothesis, scoring_config)
    speech_words = normalize_words(speech_reference, scoring_config)
    lyric_words = normalize_words(lyric_reference, scoring_config)
    speech_capacity = Counter(speech_words)
    lyric_capacity = Counter(lyric_words)
    common_vocabulary = set(speech_capacity) & set(lyric_capacity)
    labels: list[dict[str, str]] = []
    counts = Counter()
    for word in hypothesis_words:
        if word in common_vocabulary:
            label = "ambiguous"
        elif speech_capacity[word] > 0:
            label = "speech"
            speech_capacity[word] -= 1
        elif lyric_capacity[word] > 0:
            label = "lyric"
            lyric_capacity[word] -= 1
        else:
            label = "other"
        counts[label] += 1
        labels.append({"word": word, "source": label})
    grounded = counts["speech"] + counts["lyric"]
    lir = counts["lyric"] / grounded if grounded else None
    scs = (counts["lyric"] - counts["speech"]) / grounded if grounded else None
    tsr = counts["speech"] / len(speech_words) if speech_words else None
    return {
        "normalized_hypothesis": hypothesis_words,
        "token_attribution": labels,
        "n_speech": counts["speech"],
        "n_lyric": counts["lyric"],
        "n_ambiguous": counts["ambiguous"],
        "n_other": counts["other"],
        "n_grounded": grounded,
        "lir": lir,
        "tsr": tsr,
        "scs": scs,
        "no_grounded_output": grounded == 0,
    }


def score_run(config: dict[str, Any], experiment: str, model_name: str) -> Path:
    source = experiment_dir(config, experiment) / "runs" / f"{model_name}.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"run model first: {source}")
    destination = experiment_dir(config, experiment) / "scores" / f"{model_name}.jsonl"
    require_lyrics = bool(config["scoring"].get("require_lyric_reference", True))
    require_crop_lyrics = bool(
        config["scoring"].get("require_crop_aligned_lyrics", True)
    )
    output: list[dict[str, Any]] = []
    external_alignments = _load_output_alignments(config, model_name)
    missing = 0
    for row in read_jsonl(source):
        if not row.get("words") and row["sample_id"] in external_alignments:
            row["words"] = external_alignments[row["sample_id"]]
        lyrics = str(row.get("lyric_reference") or "")
        if not lyrics:
            missing += 1
            score = _empty_score("missing lyric reference")
        elif require_crop_lyrics and row.get("lyric_reference_scope") != "crop":
            missing += 1
            score = _empty_score("lyric reference is not crop-aligned")
        elif row.get("error"):
            score = _empty_score("inference error")
        else:
            score = attribute_tokens(
                str(row.get("hyp", "")),
                str(row.get("speech_reference", "")),
                lyrics,
                config["scoring"],
            )
            score["score_error"] = None
        output.append({**row, **score})
    if require_lyrics and missing:
        raise RuntimeError(
            f"{missing}/{len(output)} rows lack a usable crop-aligned lyric reference; "
            "confirmation scoring aborted"
        )
    write_jsonl(destination, output)
    return destination


def _load_output_alignments(
    config: dict[str, Any], model_name: str
) -> dict[str, list[dict[str, Any]]]:
    path = config["models"].get(model_name, {}).get("word_alignment_manifest")
    if not path or str(path).startswith("__SET_ME_"):
        return {}
    output: dict[str, list[dict[str, Any]]] = {}
    for row in load_json_or_jsonl(path):
        if "sample_id" not in row or "words" not in row:
            raise ValueError("output alignment records need sample_id and words")
        output[str(row["sample_id"])] = list(row["words"])
    return output


def _empty_score(reason: str) -> dict[str, Any]:
    return {
        "normalized_hypothesis": [],
        "token_attribution": [],
        "n_speech": 0,
        "n_lyric": 0,
        "n_ambiguous": 0,
        "n_other": 0,
        "n_grounded": 0,
        "lir": None,
        "tsr": None,
        "scs": None,
        "no_grounded_output": True,
        "score_error": reason,
    }


def score_time_window(
    hypothesis_words: list[dict[str, Any]] | None,
    start_s: float,
    end_s: float,
    speech_reference: str,
    lyric_reference: str,
    scoring_config: dict[str, Any],
    *,
    speech_words: list[dict[str, Any]] | None = None,
    lyric_words: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if not hypothesis_words:
        return None
    selected = [
        str(item.get("word", ""))
        for item in hypothesis_words
        if float(item.get("end", -1)) > start_s and float(item.get("start", 1e9)) < end_s
    ]
    if speech_words is not None:
        speech_reference = " ".join(
            str(item.get("word", ""))
            for item in speech_words
            if float(item.get("end", -1)) > start_s
            and float(item.get("start", 1e9)) < end_s
        )
    if lyric_words is not None:
        lyric_reference = " ".join(
            str(item.get("word", ""))
            for item in lyric_words
            if float(item.get("end", -1)) > start_s
            and float(item.get("start", 1e9)) < end_s
        )
    return attribute_tokens(
        " ".join(selected), speech_reference, lyric_reference, scoring_config
    )
