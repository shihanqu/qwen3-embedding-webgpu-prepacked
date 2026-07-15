#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${LLAMA_QUANTIZE:-}" ]]; then
  echo "Set LLAMA_QUANTIZE to a llama.cpp llama-quantize executable." >&2
  exit 1
fi

mkdir -p models
source_model="${QWEN3_F16_PATH:-models/Qwen3-Embedding-0.6B-f16.gguf}"
output_model="${QWEN3_Q40_PATH:-models/qwen3-embedding-0.6b-q4_0.gguf}"
url="${QWEN3_F16_URL:-https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-f16.gguf}"

if [[ ! -f "$source_model" ]]; then
  curl --fail --location --continue-at - --progress-bar "$url" --output "$source_model"
fi

"$LLAMA_QUANTIZE" "$source_model" "$output_model" Q4_0
echo "Model ready: $output_model"
