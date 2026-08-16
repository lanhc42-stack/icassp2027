#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-/root/autodl-tmp/icassp2027}"
cd "$project_root"

export PYTHONPATH="$project_root/src"
python_bin="$project_root/.venv/bin/python"
config="$project_root/configs/e7.local.yaml"
log_dir="$project_root/logs"
mkdir -p "$log_dir"

while [[ ! -f "$log_dir/e5.COMPLETE" ]]; do
  sleep 15
done
"$python_bin" -m source_capture.e7 all --config "$config"

date --iso-8601=seconds > "$log_dir/e7.COMPLETE"
