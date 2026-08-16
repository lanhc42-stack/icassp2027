from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


PLACEHOLDER_PREFIX = "__SET_ME_"


def load_config(path: str | Path, *, require_paths: bool = True) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    _resolve_known_paths(config)
    if require_paths:
        placeholders = find_placeholders(config)
        if placeholders:
            rendered = "\n".join(f"  - {key}: {value}" for key, value in placeholders)
            raise ValueError(
                "configuration still contains placeholders:\n"
                f"{rendered}\nCopy the placeholder config and fill these values."
            )
    return config


def _resolve_known_paths(config: dict[str, Any]) -> None:
    base = Path(config["_config_dir"])
    for section, key in (
        ("project", "output_root"),
        ("datasets", "librispeech_test_clean"),
        ("datasets", "musdb18hq_root"),
        ("datasets", "lyrics_manifest"),
        ("datasets", "speech_alignment_manifest"),
        ("models.whisper", "model_path"),
        ("models.ctc", "word_alignment_manifest"),
    ):
        node: Any = config
        parts = section.split(".")
        for part in parts:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        value = node.get(key) if isinstance(node, dict) else None
        if not isinstance(value, str) or value.startswith(PLACEHOLDER_PREFIX):
            continue
        expanded = Path(value).expanduser()
        if not expanded.is_absolute():
            expanded = (base / expanded).resolve()
        node[key] = str(expanded)


def find_placeholders(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).startswith("_"):
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(find_placeholders(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_placeholders(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and value.startswith(PLACEHOLDER_PREFIX):
        found.append((prefix, value))
    return found


def output_root(config: dict[str, Any]) -> Path:
    return Path(config["project"]["output_root"])


def experiment_dir(config: dict[str, Any], experiment: str) -> Path:
    if experiment not in {"e1", "e2", "e3"}:
        raise ValueError(f"unknown experiment: {experiment}")
    return output_root(config) / experiment


def snapshot_config(config: dict[str, Any]) -> Path:
    destination = output_root(config) / "config.snapshot.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    destination.write_text(
        yaml.safe_dump(clean, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return destination
