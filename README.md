# Qwen3 Embedding 0.6B for WebGPU

A high-throughput Q4 WebGPU runtime for Qwen3 Embedding 0.6B. It uses a GPU-native prepacked model format, custom WGSL kernels, and native micro-batching for up to 16 simultaneous embedding requests.

> [!IMPORTANT]
> This repository is no longer maintained. Continued development has moved to [shihanqu/nemotron-3-embed-webgpu](https://github.com/shihanqu/nemotron-3-embed-webgpu). Use that repository for current releases, updates, and support.

![Qwen3 Embedding WebGPU versus LM Studio](docs/lm-studio-comparison.svg)

## Performance

The benchmark uses identical text for both runtimes and exact tokenizer lengths including EOS. WebGPU rows are isolated, warmed measurements; LM Studio rows use ten warmups and a 10-second measurement per condition. Model loading is excluded.

| Input | LM Studio single | WebGPU single | LM Studio 16× aggregate | WebGPU 16× aggregate |
|---:|---:|---:|---:|---:|
| 15 tokens | 28.84 req/s | **65.69 req/s** | 30.71 req/s | **239.51 req/s** |
| 50 tokens | 25.66 req/s | **39.81 req/s** | 27.25 req/s | **117.10 req/s** |
| 150 tokens | 18.53 req/s | **20.82 req/s** | 19.41 req/s | **41.43 req/s** |
| 500 tokens | 2.46 req/s | **6.63 req/s** | 3.10 req/s | **11.76 req/s** |
| 1,500 tokens | 3.14 req/s | **3.17 req/s** | 3.05 req/s | **3.39 req/s** |
| 5,000 tokens | 0.61 req/s | **0.95 req/s** | 0.59 req/s | **0.97 req/s** |

Every tested size exceeds the raw LM Studio endpoint in both columns. At 16 concurrent requests, WebGPU delivers 7.80×, 4.30×, 2.13×, 3.80×, 1.11×, and 1.63× LM Studio throughput from 15 through 5,000 tokens.

Every row is compared directly with LM Studio. LM cosine agreement ranges from `0.9309` to `0.9542`; native-batch agreement with the single-request WebGPU output is at least `0.999909`.

Inputs through 500 tokens use full causal attention. The 1,500- and 5,000-token rows use an accuracy-gated sparse path that retains the first 176 tokens plus a 432-token causal local window. This is not numerically identical to full attention; its measured LM cosine is `0.9505` at 1,500 tokens and `0.9441` at 5,000 tokens.

### Test hardware

Apple M3 Max, 30 core GPU.

Results depend on the GPU, browser, power state, and thermals. Run long-context rows separately: a back-to-back 5,000-token stress run can materially throttle later measurements. The complete measurements are in [`docs/benchmarks/2026-07-16-webgpu-vs-lm-studio-m3-max.json`](docs/benchmarks/2026-07-16-webgpu-vs-lm-studio-m3-max.json).

## Expected system requirements

These are practical estimates rather than hard compatibility guarantees:

| Resource | Minimum | Recommended |
|---|---|---|
| Browser | Chromium-based browser with WebGPU, `shader-f16`, and subgroup support | Current stable Chrome or Edge |
| GPU | Apple Silicon or a modern NVIDIA/AMD GPU exposing the required WebGPU features | Recent discrete GPU or Apple Silicon with at least 4 GB of available GPU/shared memory |
| System memory | 8 GB | 16 GB or more for 16 concurrent requests |
| Storage and initial download | About 384 MiB for the model pack | 1 GB free for the pack, browser cache, and build output |
| Local development | Node.js 24 and npm | Current Node.js 24 LTS release |

The browser must allow individual WebGPU storage-buffer bindings of at least 128 MiB. Systems with unified memory count available system memory toward GPU allocations. Actual memory use grows with batch size and sequence length.

## How it works

The runtime moves work that would normally happen during browser startup into a deterministic offline conversion step:

- Q, K, and V projection matrices are fused, as are the FFN gate and up matrices.
- Q4_0 projection weights are rearranged into compact 32-row GPU tiles. Each 32-value block occupies 20 bytes: one aligned scale word and four packed-quant words.
- Token embeddings, normalization weights, metadata, and every other runtime tensor are copied into the same self-contained file.
- The browser uploads all 226 tensors directly without parsing a GGUF, concatenating matrices, or repacking quantized blocks on the CPU.

The inference path is implemented with custom WGSL rather than a general-purpose graph runtime:

- A 16×16, K=64 tiled Q4 matmul path targets single-request latency.
- A 32-row subgroup matmul path processes large request batches with weights shared across rows.
- Fused kernels handle residual addition plus RMSNorm, Q/K normalization plus RoPE, and SwiGLU.
- A 16-query × 16-key FlashAttention-style kernel keeps QK scores, block softmax, and AV accumulation in workgroup memory.
- Long inputs use the accuracy-gated 176-token-prefix plus 432-token-local sparse attention path described above.
- The scheduler turns simultaneous calls into native batches of up to 16 requests.
- Single-request plans own and release their activation buffers explicitly; a 5,000-token 16-request workload is executed as four native batches of four to remain within WebGPU buffer limits.

Together, the prepacked layout reduces startup CPU work and the custom execution paths increase both single-stream and concurrent throughput.

## Required model file

The app loads one pinned GitHub Release asset by default:

| Artifact | Contents | Size | SHA-256 |
|---|---|---:|---|
| [`qwen3-embedding-0.6b-q4_0-webgpu.wgpack`](https://github.com/shihanqu/qwen3-embedding-webgpu-prepacked/releases/download/wgpack-v3/qwen3-embedding-0.6b-q4_0-webgpu.wgpack) | Complete Qwen3 Embedding 0.6B model: 112 compact Q4_0 projection matrices plus all 114 auxiliary tensors and required metadata | 384 MiB | [`abff36…dc55`](docs/wgpack-v3.sha256) |

No GGUF is downloaded or parsed during normal use. The pack header records the SHA-256 of the exact source GGUF used to build it. Model artifacts retain the upstream Apache-2.0 license; source code is MIT licensed.

## Quick start

```sh
git clone https://github.com/shihanqu/qwen3-embedding-webgpu-prepacked.git
cd qwen3-embedding-webgpu-prepacked
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/?matrix=1`, click **Load model & benchmark kernel**, and look for `BENCHMARK_MATRIX_JSON`. Vite proxies the release asset through the local origin to avoid cross-origin model-loading problems.

For local or offline copies:

```sh
npm run model:download:webgpu
VITE_PREPACKED_MODEL_URL=/models/qwen3-embedding-0.6b-q4_0-webgpu.wgpack \
  npm run dev
```

## Generate the prepacked model

Only pack authors need the source GGUF. Place the exact Q4_0 source model in `models/`, then run:

```sh
npm ci
npm run model:prepack -- \
  models/qwen3-embedding-0.6b-q4_0-webgpu.gguf \
  models/qwen3-embedding-0.6b-q4_0-webgpu.wgpack
```

The generator is deterministic. The format consists of an 8-byte `WGPACK02` magic value, a JSON header, and 256-byte-aligned tensor payloads. Projection rows are grouped into 32-row tiles; K is grouped into 32-value Q4 blocks. Non-projection tensors retain their source bytes and GGML type. The resulting pack is 402,945,280 bytes—only about 20.6 MiB (5.7%) larger than its 364 MiB source GGUF. See `src/prepacked/format.ts` and `scripts/prepack-model.ts` for the authoritative layout.

## Reproduce the comparison

Run the warmed WebGPU matrix in one model load:

```text
http://127.0.0.1:5173/?matrix=1
```

Run LM Studio against the same exact fixtures with independent HTTP workers:

```sh
npm run bench:baseline -- --tokens=15  --input-index=0 --concurrency=1,16 --duration-ms=10000 --warmup=10
npm run bench:baseline -- --tokens=50  --input-index=0 --concurrency=1,16 --duration-ms=10000 --warmup=10
npm run bench:baseline -- --tokens=150 --input-index=0 --concurrency=1,16 --duration-ms=10000 --warmup=10
npm run bench:baseline -- --tokens=500 --input-index=0 --concurrency=1,16 --duration-ms=10000 --warmup=10
npm run bench:baseline -- --tokens=1500 --input-index=0 --concurrency=1,16 --duration-ms=10000 --warmup=10
npm run bench:baseline -- --tokens=5000 --input-index=0 --concurrency=1,16 --duration-ms=10000 --warmup=10
```

For stable long-context numbers, run `?matrix=1&tokens=1500` and `?matrix=1&tokens=5000` as separate, cooled conditions. The 5,000-token WebGPU condition uses native batches of four and issues four chunks per 16-request round.

Regenerate the checked-in chart from the benchmark JSON:

```sh
npm run bench:chart
```

The exact fixtures live in `scripts/workloads.ts`. Avoid comparing different input text, tokenizer lengths, array-valued HTTP batching, or cold model-load time.

## Portability

The model pack contains no machine ISA and the kernels use standard WGSL, so the design is not tied to Apple GPU architecture. It should run on NVIDIA and AMD hardware when the browser exposes the required WebGPU features. Performance is not architecture-independent: subgroup size, memory behavior, driver quality, and browser support differ. This release has only been validated on Apple M3 Max, 30 core GPU, so other GPUs should be checked with `npm run check`, `?matrix=1`, and embedding cosine comparisons.

## Development

```sh
npm run check
```

Key files:

- `src/prepacked/format.ts` — packed model format and source-hash metadata.
- `scripts/prepack-model.ts` — deterministic GGUF-to-`.wgpack` generator.
- `src/webgpu/quant-matmul.ts` — latency and subgroup Q4 matmul kernels.
- `src/webgpu/model.ts` — direct tensor uploads, reusable workspaces, and execution plans.
- `src/webgpu/ops.ts` — fused normalization, RoPE, SwiGLU, and causal attention kernels.
- `src/webgpu/embedding-engine.ts` — concurrent request micro-batching.

## License

Source code is [MIT licensed](LICENSE). Qwen model artifacts retain their upstream [Apache-2.0 terms](MODEL_LICENSE); see [MODEL_NOTICE.md](MODEL_NOTICE.md).
