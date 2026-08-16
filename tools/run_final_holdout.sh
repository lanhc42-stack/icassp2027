#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-/root/autodl-tmp/icassp2027}"
cd "$project_root"

export PYTHONPATH="$project_root/src"
export QWEN_ASR_MODEL="$project_root/models/Qwen3-ASR-1.7B"
export QWEN_ALIGNER_MODEL="$project_root/models/Qwen3-ForcedAligner-0.6B"
python_bin="$project_root/.venv/bin/python"
qwen_python="$project_root/qwen-venv/bin/python"
base_config="$project_root/configs/e1_e3.local.yaml"
log_dir="$project_root/logs"
mkdir -p "$log_dir"

if [[ ! -f "$project_root/docs/analysis-freeze-2026-08-15.md" ]]; then
  echo "refusing to open holdout without the archived analysis freeze" >&2
  exit 1
fi
while [[ ! -f "$log_dir/e8.COMPLETE" ]]; do
  sleep 15
done

run_and_score() {
  local experiment=$1
  local model=$2
  "$python_bin" -m source_capture run \
    --config "$base_config" \
    --experiment "$experiment" \
    --model "$model" \
    --split holdout
  "$python_bin" -m source_capture score \
    --config "$base_config" \
    --experiment "$experiment" \
    --model "$model"
  "$python_bin" -m source_capture analyze \
    --config "$base_config" \
    --experiment "$experiment"
}

for experiment in e1 e3 e2; do
  run_and_score "$experiment" whisper
done

"$qwen_python" "$project_root/tools/qwen_asr_server.py" \
  > "$log_dir/qwen_holdout_server.log" 2>&1 &
qwen_server_pid=$!
cleanup() {
  kill "$qwen_server_pid" 2>/dev/null || true
  wait "$qwen_server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
  if grep -q "serving on" "$log_dir/qwen_holdout_server.log" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$qwen_server_pid" 2>/dev/null; then
    echo "Qwen server exited during startup" >&2
    exit 1
  fi
  sleep 1
done
if ! grep -q "serving on" "$log_dir/qwen_holdout_server.log" 2>/dev/null; then
  echo "Qwen server did not become ready" >&2
  exit 1
fi

for experiment in e1 e3 e2; do
  run_and_score "$experiment" qwen3
done
cleanup
trap - EXIT INT TERM

"$python_bin" -m source_capture.e4 prepare --config configs/e4_holdout.local.yaml --experiment all
"$python_bin" -m source_capture.e4 run --config configs/e4_holdout.local.yaml --experiment all
"$python_bin" -m source_capture.e4 analyze --config configs/e4_holdout.local.yaml --experiment all

"$python_bin" -m source_capture.e5 prepare --config configs/e5_holdout.local.yaml
"$python_bin" -m source_capture.e5 run --config configs/e5_holdout.local.yaml
"$python_bin" -m source_capture.e5 analyze --config configs/e5_holdout.local.yaml
"$python_bin" -m source_capture.e7 prepare-holdout --config configs/e7.local.yaml
"$python_bin" -m source_capture.e7 evaluate-holdout --config configs/e7.local.yaml
"$python_bin" -m source_capture.e8 all --config configs/e8_holdout.local.yaml

date --iso-8601=seconds > "$log_dir/final_holdout.COMPLETE"
