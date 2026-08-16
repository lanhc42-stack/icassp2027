from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyze import analyze_experiment
from .config import find_placeholders, load_config
from .generate import generate_experiment
from .prepare import prepare_pairs
from .run import run_model
from .scoring import score_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m source_capture",
        description="E1-E3 source-capture confirmation experiment pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "prepare", "generate", "run", "score", "analyze"):
        item = subparsers.add_parser(name)
        item.add_argument("--config", required=True)
        if name in {"generate", "run", "score", "analyze"}:
            item.add_argument(
                "--experiment",
                required=True,
                choices=["e1", "e2", "e3", "all"] if name == "generate" else ["e1", "e2", "e3"],
            )
        if name in {"run", "score"}:
            item.add_argument("--model", required=True, choices=["whisper", "qwen3", "ctc"])
        if name == "run":
            item.add_argument("--overwrite", action="store_true")
            item.add_argument(
                "--split", choices=["development", "holdout", "all"], default="development"
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        config = load_config(args.config, require_paths=False)
        placeholders = find_placeholders(config)
        if placeholders:
            print("Configuration structure is valid. Fill these placeholders:")
            for key, value in placeholders:
                print(f"  {key}: {value}")
            return 2
        _validate_existing_paths(config)
        print("Configuration is valid and all local paths exist.")
        return 0

    config = load_config(args.config, require_paths=True)
    if args.command == "prepare":
        print(prepare_pairs(config))
    elif args.command == "generate":
        experiments = ["e1", "e2", "e3"] if args.experiment == "all" else [args.experiment]
        for experiment in experiments:
            print(generate_experiment(config, experiment))
    elif args.command == "run":
        print(
            run_model(
                config,
                args.experiment,
                args.model,
                overwrite=args.overwrite,
                split=args.split,
            )
        )
    elif args.command == "score":
        print(score_run(config, args.experiment, args.model))
    elif args.command == "analyze":
        for path in analyze_experiment(config, args.experiment):
            print(path)
    return 0


def _validate_existing_paths(config: dict) -> None:
    paths = [
        config["datasets"]["librispeech_test_clean"],
        config["datasets"]["musdb18hq_root"],
        config["datasets"]["lyrics_manifest"],
        config["datasets"]["speech_alignment_manifest"],
        config["models"]["whisper"]["model_path"],
        config["models"]["ctc"].get("word_alignment_manifest"),
    ]
    missing = [str(path) for path in paths if path and not Path(path).exists()]
    if missing:
        raise FileNotFoundError("configured local paths do not exist:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    sys.exit(main())
