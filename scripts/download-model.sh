#!/usr/bin/env bash
set -euo pipefail

mkdir -p models
url="${QWEN3_GGUF_URL:-https://huggingface.co/endyjasmi/Qwen3-Embedding-0.6B-Q4_K_M-GGUF/resolve/main/qwen3-embedding-0.6b-q4_k_m.gguf}"
out="${QWEN3_GGUF_PATH:-models/qwen3-embedding-0.6b-q4_k_m.gguf}"

curl --fail --location --continue-at - --progress-bar "$url" --output "$out"
echo "Model ready: $out"

