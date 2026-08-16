#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/data/downloads"
parts="$out/nus48e.parts"
mkdir -p "$parts"

url='https://zenodo.org/api/records/19595152/files/nus-smc-corpus_48.zip/content'
resolve='zenodo.org:443:137.138.52.235'
size=1080034316
nparts=8
chunk=$(( (size + nparts - 1) / nparts ))

download_part() {
  local idx="$1"
  local start=$((idx * chunk))
  local end=$((start + chunk - 1))
  if (( end >= size )); then end=$((size - 1)); fi
  local path="$parts/part_$(printf '%02d' "$idx")"
  local expected=$((end - start + 1))
  if [[ -f "$path" ]] && [[ "$(stat -f '%z' "$path")" -eq "$expected" ]]; then
    return 0
  fi
  curl --resolve "$resolve" -L --fail --retry 8 --retry-delay 3 \
    --range "$start-$end" -o "$path" "$url"
  [[ "$(stat -f '%z' "$path")" -eq "$expected" ]]
}

export -f download_part
export root out parts url resolve size nparts chunk
seq 0 $((nparts - 1)) | xargs -n1 -P4 -I{} bash -c 'download_part "$1"' _ {}

for i in $(seq 0 $((nparts - 1))); do
  idx="$(printf '%02d' "$i")"
  expected=$chunk
  if [[ "$idx" == "$(printf '%02d' $((nparts - 1)))" ]]; then
    expected=$((size - chunk * (nparts - 1)))
  fi
  [[ "$(stat -f '%z' "$parts/part_$idx")" -eq "$expected" ]]
done

cat "$parts"/part_* > "$out/nus-smc-corpus_48.zip"
[[ "$(stat -f '%z' "$out/nus-smc-corpus_48.zip")" -eq "$size" ]]
echo "Downloaded $out/nus-smc-corpus_48.zip"
