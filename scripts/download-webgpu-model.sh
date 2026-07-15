#!/usr/bin/env bash
set -euo pipefail

filename="qwen3-embedding-0.6b-q4_0-webgpu.gguf"
url="${QWEN3_WEBGPU_MODEL_URL:-https://github.com/shihanqu/qwen3-embedding-webgpu-prepacked/releases/download/prepacked-v1/$filename}"
out="${QWEN3_WEBGPU_MODEL_PATH:-models/$filename}"
expected_sha256="4acbfc4947344ca4d4a215ee35e601c5e6f505172b517da194460e2ff113433e"

mkdir -p "$(dirname "$out")"

if [[ -f "$out" ]]; then
  existing_sha256="$(shasum -a 256 "$out" | awk '{print $1}')"
  if [[ "$existing_sha256" == "$expected_sha256" ]]; then
    echo "Model already present and verified: $out"
    exit 0
  fi
fi

curl --fail --location --continue-at - --progress-bar "$url" --output "$out"

actual_sha256="$(shasum -a 256 "$out" | awk '{print $1}')"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "SHA-256 mismatch for $out" >&2
  echo "expected: $expected_sha256" >&2
  echo "actual:   $actual_sha256" >&2
  exit 1
fi

echo "Model ready: $out"
