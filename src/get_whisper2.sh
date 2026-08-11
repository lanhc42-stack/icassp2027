#!/bin/bash
# huggingface_hub refuses hf-mirror's 308 redirect, so fetch the files directly
# with curl -L. Only what transformers needs to load the model.
set -u
BASE=https://hf-mirror.com/openai/whisper-large-v3/resolve/main
DST=/opt/tiger/icassp2027/models/whisper-large-v3
mkdir -p "$DST"

FILES="config.json generation_config.json preprocessor_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt normalizer.json special_tokens_map.json added_tokens.json model.safetensors"

for f in $FILES; do
    if [ -s "$DST/$f" ]; then echo "have $f"; continue; fi
    code=$(curl -sL -w '%{http_code}' --max-time 1800 -o "$DST/$f" "$BASE/$f")
    sz=$(stat -c %s "$DST/$f" 2>/dev/null || echo 0)
    echo "$f -> HTTP $code  $((sz/1024)) KB"
    if [ "$code" != "200" ]; then rm -f "$DST/$f"; fi
done

echo "=== result ==="
ls -la "$DST"
du -sh "$DST"
