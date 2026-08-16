#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-/root/autodl-tmp/icassp2027}"
cd "$project_root"

export PYTHONPATH="$project_root/src"
python_bin="$project_root/.venv/bin/python"
config="$project_root/configs/e8.local.yaml"
binary="$project_root/tools/deepfilternet/deep-filter"
log_dir="$project_root/logs"
mkdir -p "$log_dir"

while [[ ! -f "$log_dir/e6.COMPLETE" ]]; do
  sleep 15
done
if [[ ! -x "$binary" ]]; then
  echo "missing executable DeepFilterNet binary: $binary" >&2
  exit 1
fi
"$python_bin" -m source_capture.e8 all --config "$config"

date --iso-8601=seconds > "$log_dir/e8.COMPLETE"
