#!/bin/bash
# Isolated python deps for the ICASSP work: installed to a --target dir so the
# shared Triton image's torch/torchaudio/numpy/DALI are never touched.
set -u
LIBS=/opt/tiger/icassp2027/pylibs
mkdir -p "$LIBS"

echo "=== mirror reachability ==="
for u in https://hf-mirror.com https://huggingface.co; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 12 "$u/openai/whisper-large-v3/resolve/main/config.json")
    echo "$u -> $code"
done

echo "=== installing to $LIBS ==="
pip3 install -q --target="$LIBS" --upgrade jiwer 2>&1 | tail -3

export PYTHONPATH="$LIBS"
python3 - <<'PY'
import jiwer, torch, torchaudio, transformers
print("jiwer", jiwer.__version__)
print("torch", torch.__version__, "torchaudio", torchaudio.__version__)
print("transformers", transformers.__version__)
PY

echo "=== qwen 1.7b download ==="
du -sh /opt/tiger/icassp2027/models/qwen3_asr_1.7b 2>/dev/null
