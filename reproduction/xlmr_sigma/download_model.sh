#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="${XLMR_MODEL_DIR:-$REPO_ROOT/.cache/models/xlm-roberta-base}"
MODEL_ID="FacebookAI/xlm-roberta-base"
REVISION="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"

if ! command -v hf >/dev/null 2>&1; then
    echo "error: hf CLI not found; install huggingface_hub first" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"
hf download "$MODEL_ID" \
    config.json \
    tokenizer_config.json \
    sentencepiece.bpe.model \
    model.safetensors \
    --revision "$REVISION" \
    --local-dir "$MODEL_DIR"

echo "Model snapshot: $MODEL_ID@$REVISION"
echo "Local directory: $MODEL_DIR"
