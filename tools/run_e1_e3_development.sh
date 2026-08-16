#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG_YAML" >&2
  exit 2
fi

config=$1
project_root=$(cd "$(dirname "$0")/.." && pwd)
log_dir="$project_root/logs"
mkdir -p "$log_dir"
cd "$project_root"
export PYTHONPATH="$project_root/src"

run_and_score() {
  local experiment=$1
  local model=$2
  "$project_root/.venv/bin/python" -m source_capture run \
    --config "$config" \
    --experiment "$experiment" \
    --model "$model" \
    --split development \
    > "$log_dir/${experiment}_${model}.log" 2>&1
  "$project_root/.venv/bin/python" -m source_capture score \
    --config "$config" \
    --experiment "$experiment" \
    --model "$model" \
    > "$log_dir/${experiment}_${model}_score.log" 2>&1
  "$project_root/.venv/bin/python" -m source_capture analyze \
    --config "$config" \
    --experiment "$experiment" \
    > "$log_dir/${experiment}_analyze.log" 2>&1
}

for experiment in e1 e3 e2; do
  run_and_score "$experiment" whisper
done

: "${QWEN_ASR_MODEL:?set QWEN_ASR_MODEL to the local Qwen3-ASR checkpoint}"
: "${QWEN_ALIGNER_MODEL:?set QWEN_ALIGNER_MODEL to the local forced-aligner checkpoint}"

"$project_root/qwen-venv/bin/python" "$project_root/tools/qwen_asr_server.py" \
  > "$log_dir/qwen_server.log" 2>&1 &
qwen_server_pid=$!
cleanup() {
  kill "$qwen_server_pid" 2>/dev/null || true
  wait "$qwen_server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
  if grep -q "serving on" "$log_dir/qwen_server.log" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$qwen_server_pid" 2>/dev/null; then
    echo "Qwen server exited during startup; see $log_dir/qwen_server.log" >&2
    exit 1
  fi
  sleep 1
done
if ! grep -q "serving on" "$log_dir/qwen_server.log" 2>/dev/null; then
  echo "Qwen server did not become ready within 120 seconds" >&2
  exit 1
fi

for experiment in e1 e3 e2; do
  run_and_score "$experiment" qwen3
done

touch "$log_dir/e1_e3_development.COMPLETE"
