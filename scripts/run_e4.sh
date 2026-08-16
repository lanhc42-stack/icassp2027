#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-/root/autodl-tmp/icassp2027}"
cd "$project_root"

export PYTHONPATH="$project_root/src"
python_bin="$project_root/.venv/bin/python"
config="$project_root/configs/e4.local.yaml"
log_dir="$project_root/logs"
mkdir -p "$log_dir"

"$python_bin" -m source_capture.e4 prepare --config "$config" --experiment all
"$python_bin" -m source_capture.e4 run --config "$config" --experiment e4a
"$python_bin" -m source_capture.e4 analyze --config "$config" --experiment e4a
"$python_bin" -m source_capture.e4 run --config "$config" --experiment e4b
"$python_bin" -m source_capture.e4 analyze --config "$config" --experiment e4b

date --iso-8601=seconds > "$log_dir/e4.COMPLETE"
