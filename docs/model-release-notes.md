# Qwen3 Embedding 0.6B self-contained WebGPU pack

This release contains the complete model artifact loaded by the WebGPU app by default. It does not require a GGUF at runtime.

- Asset: `qwen3-embedding-0.6b-q4_0-webgpu.wgpack`
- Contents: all 226 runtime tensors and required metadata; 112 Q4_0 projection matrices use the compact GPU tile layout
- Fused projections: Q, K, and V are stored as one matrix per layer; FFN gate and up are stored as one matrix per layer
- Source GGUF SHA-256: `4acbfc4947344ca4d4a215ee35e601c5e6f505172b517da194460e2ff113433e`
- Size: 402,945,280 bytes (384 MiB)
- SHA-256: `abff362389e436e7fff44cb68bef3948cb81e47e40d9680635f7894acb90dc55`
- Model license: Apache License 2.0

See `MODEL_NOTICE.md` and `MODEL_LICENSE` in the tagged source, which are also attached to this release. The repository source code remains under its separate MIT license.
