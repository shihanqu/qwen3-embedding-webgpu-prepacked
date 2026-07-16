#!/usr/bin/env bash
set -euo pipefail

filename="qwen3-embedding-0.6b-q4_0-webgpu.wgpack"
url="${QWEN3_WEBGPU_MODEL_URL:-https://github.com/shihanqu/qwen3-embedding-webgpu-prepacked/releases/download/wgpack-v2/$filename}"
out="${QWEN3_WEBGPU_MODEL_PATH:-models/$filename}"
expected_sha256="4678826b4e9ab225dedd45c930b5f5e2db0aa6b919239c5404c23524156f04ce"

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
