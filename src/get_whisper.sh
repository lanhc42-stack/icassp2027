#!/bin/bash
set -u
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH=/opt/tiger/icassp2027/pylibs
MODELS=/opt/tiger/icassp2027/models

echo "=== qwen3_asr_1.7b config ==="
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/opt/tiger/icassp2027/models/qwen3_asr_1.7b/config.json")
if p.exists():
    c = json.loads(p.read_text())
    print("model_type:", c.get("model_type"))
    print("architectures:", c.get("architectures"))
    for k in ("audio_config", "text_config"):
        v = c.get(k)
        if isinstance(v, dict):
            print(f"  {k}: model_type={v.get('model_type')} layers={v.get('num_hidden_layers')} hidden={v.get('hidden_size')}")
    import transformers
    name = (c.get("architectures") or ["?"])[0]
    print("supported by transformers:", hasattr(transformers, name))
else:
    print("config not present yet")
PY

echo "=== downloading whisper-large-v3 via mirror ==="
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    "openai/whisper-large-v3",
    local_dir="/opt/tiger/icassp2027/models/whisper-large-v3",
    allow_patterns=["*.json", "*.txt", "model.safetensors", "*.model"],
    max_workers=4,
)
print("downloaded to", p)
PY

du -sh "$MODELS"/* 2>/dev/null
