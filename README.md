# Qwen3 Embedding 0.6B on WebGPU

A complete, browser-native Qwen3 Embedding 0.6B runtime built from custom WGSL kernels. It loads a GGUF model directly, runs all 28 transformer layers on WebGPU, and micro-batches up to 16 simultaneous embedding requests.

The implementation does **not** delegate inference to ONNX Runtime. Quantized projection, embedding lookup, RMSNorm/residual, Q/K normalization and RoPE, causal attention, SwiGLU, and last-token L2 normalization are implemented in this repository.

![WebGPU and LM Studio benchmark comparison](docs/lm-studio-comparison.svg)

## Required model

The optimized path requires this exact generated file:

| Property | Value |
|---|---|
| Runtime path | `models/qwen3-embedding-0.6b-q4_0.gguf` |
| Quantization | GGUF Q4_0; token embeddings remain Q6_K, as selected by llama.cpp |
| Source | [Qwen/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF), file `Qwen3-Embedding-0.6B-f16.gguf` |
| Generated size | 381,335,744 bytes (364 MiB) |
| SHA-256 | `4acbfc4947344ca4d4a215ee35e601c5e6f505172b517da194460e2ff113433e` |

Generate Q4_0 directly from Qwen's f16 model. Do not re-quantize an existing Q4 model: double quantization materially reduces embedding agreement. The optional `model:download:q4km` script downloads a higher-quality Q4_K_M model for experiments, but that is not the model used by the published acceptance numbers.

The model weights are not committed to GitHub. Qwen's model is licensed separately under Apache-2.0.

## Quick start

### 1. Prerequisites

- Node.js 24 or newer and npm.
- Chrome, Edge, or another Chromium browser with WebGPU, `shader-f16`, and subgroup support enabled.
- [`llama-quantize`](https://github.com/ggml-org/llama.cpp) from a recent llama.cpp build.
- About 1.7 GB of free disk space while preparing the model; the final Q4_0 file is about 364 MiB.

On macOS, one way to build the required quantizer is:

```sh
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DLLAMA_METAL=ON
cmake --build llama.cpp/build --config Release -j
```

The executable will normally be at `llama.cpp/build/bin/llama-quantize`.

### 2. Install and prepare the model

```sh
git clone https://github.com/shihanqu/qwen3-embedding-webgpu.git
cd qwen3-embedding-webgpu
npm ci

LLAMA_QUANTIZE=/absolute/path/to/llama-quantize \
  npm run model:prepare:q40
```

The preparation script downloads the official 1.2 GB f16 GGUF if it is absent, then writes the required Q4_0 file under `models/`. You can provide an existing f16 download with `QWEN3_F16_PATH=/path/to/model.gguf`.

Verify the generated model if you want to reproduce the tested artifact exactly:

```sh
shasum -a 256 models/qwen3-embedding-0.6b-q4_0.gguf
```

### 3. Start the browser runtime

```sh
npm run dev
```

Open `http://127.0.0.1:5173/?q40=1&scheduler=1` and select **Load model & benchmark kernel**. The first load reads 364 MiB from disk, uploads the weights, downloads Qwen's tokenizer through Transformers.js, and compiles the WebGPU pipelines. Later executions reuse the compiled plans.

LM Studio is only required for comparison benchmarks. The WebGPU model itself runs entirely in the browser after its model and tokenizer have loaded.

## Request API

`Qwen3EmbeddingEngine.embed()` is the request-facing API. Calls made during the same scheduling window are padded after EOS, combined into a batch of at most 16, executed with a cached plan, and resolved in their original order.

```ts
import { AutoTokenizer } from '@huggingface/transformers';
import { GGUFReader, QWEN3_METADATA_KEYS } from './src/gguf/reader.ts';
import { requestWebGPUDevice } from './src/webgpu/device.ts';
import { Qwen3EmbeddingEngine, type TokenizerLike } from './src/webgpu/embedding-engine.ts';
import { Qwen3WebGPUModel } from './src/webgpu/model.ts';

const { device } = await requestWebGPUDevice();
const bytes = await fetch('/models/qwen3-embedding-0.6b-q4_0.gguf').then((response) => response.arrayBuffer());
const gguf = new GGUFReader(bytes).parse({ metadataKeys: QWEN3_METADATA_KEYS });
const runtime = new Qwen3WebGPUModel(device, gguf);
const tokenizer = await AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B');
const embeddings = new Qwen3EmbeddingEngine(runtime, tokenizer as unknown as TokenizerLike, 16);

const one = await embeddings.embed('A semantic search query');
const sixteen = await Promise.all(
  Array.from({ length: 16 }, (_, index) => embeddings.embed(`Query ${index}`)),
);
```

Each result is a normalized `Float32Array(1024)`.

## Reproducing the LM Studio comparison

1. In LM Studio, load the embedding model whose API identifier is exactly `text-embedding-qwen3-embedding-0.6b`.
2. Start LM Studio's OpenAI-compatible local server on `http://127.0.0.1:1234`.
3. Prepare the clean Q4_0 model and start this project with `npm run dev`.
4. Open one of the benchmark URLs below and press the button.

If LM Studio is on another host or port, start Vite with an override:

```sh
LM_STUDIO_URL=http://127.0.0.1:1234 npm run dev
```

Benchmark modes:

- `?q40=1&scheduler=1` — end-to-end acceptance through 16 simultaneous `embed()` calls.
- `?q40=1&sweep=1` — warmed 1/2/4/8/16 sweep at 6, 17, 26, and 105 exact tokenizer tokens.
- `?q40=1&profile=16` — per-stage GPU timestamps for a 16-request batch.
- `?q40=1` — full correctness check and the 105-token stress case.

The default correctness page can optionally compare against llama.cpp running the exact same Q4_0 file:

```sh
llama-server \
  -m models/qwen3-embedding-0.6b-q4_0.gguf \
  --embedding --pooling last -ngl 99 \
  --host 127.0.0.1 --port 1236
```

Set `Q4_0_REFERENCE_URL` when that server uses a different address.

## Published benchmark

Tests were run July 14, 2026 on:

| Component | Tested configuration |
|---|---|
| Computer | MacBook Pro, model identifier Mac15,10 |
| SoC | Apple M3 Max |
| CPU | 14 cores: 10 performance and 4 efficiency |
| GPU | 30-core Apple GPU |
| Memory | 96 GB unified memory |
| OS | macOS 26.5.2, build 25F84 |
| Browser API | Chromium WebGPU with `shader-f16`, subgroups, and timestamp queries |
| llama.cpp | Build b10015 for quantization and exact-model validation |
| WebGPU model | Clean f16 → Q4_0 conversion described above |
| LM Studio model | `text-embedding-qwen3-embedding-0.6b` at `127.0.0.1:1234` |

| Exact tokens | WebGPU single | LM Studio single | Improvement | WebGPU batch 16 aggregate | Scaling |
|---:|---:|---:|---:|---:|---:|
| 6 (acceptance) | 57.33 req/s | 31.49 req/s | **1.82×** | 461.03 req/s | **8.04×** |
| 17 | 51.06 req/s | 26.79 req/s | 1.91× | 186.22 req/s | 3.65× |
| 26 | 49.44 req/s | 24.35 req/s | 2.03× | 130.54 req/s | 2.64× |
| 105 | 25.65 req/s | 19.00 req/s | 1.35× | 38.41 req/s | 1.50× |

The declared 6-token acceptance workload clears both original goals: single-stream throughput is at least 30% higher than LM Studio, and 16 simultaneous requests deliver at least 4× the single-stream aggregate throughput.

Scaling is length-dependent. The 17-, 26-, and 105-token stress cases do not currently clear the 4× concurrency goal. They are published alongside the acceptance result so the short-query result is not mistaken for length-independent behavior.

The WebGPU output agrees with llama.cpp running the exact same clean Q4_0 GGUF at cosine `0.999995`. Batch output agrees with the single path at cosine `1.000000` for the acceptance case. Cosine against a separately quantized LM Studio model is not an implementation correctness comparison.

Raw chart data is checked in at [`docs/benchmarks/2026-07-14-m3-max.json`](docs/benchmarks/2026-07-14-m3-max.json). Regenerate the README graph with `npm run bench:chart`.

## Benchmark methodology

- Record exact Qwen tokenizer length, including EOS; do not rely on word-count estimates.
- Separate cold model download, GPU upload, and pipeline compilation from warmed inference.
- Use identical text and sequential requests for the LM Studio single-stream baseline.
- Sweep concurrency 1, 2, 4, 8, and 16 instead of reporting only the best batch.
- Report both batch latency and aggregate requests per second.
- Compare each batch embedding against the single-request output.
- Keep sequence-length results separate because GPU occupancy and attention cost change substantially with length.

## Deployment notes

- Host `models/qwen3-embedding-0.6b-q4_0.gguf` at the same path expected by the app, or update the fetch URL.
- Preserve the `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` response headers from `vite.config.ts`.
- The serving browser must expose WebGPU and `shader-f16`; the app fails early with a useful error when these are unavailable.
- The current app is a development and benchmark console. Integrate `Qwen3EmbeddingEngine` into your own request or UI layer for production use.

## Development

```sh
npm ci
npm run check
npm run bench:chart
```

GitHub Actions runs the unit tests and production build on every push and pull request. Unit coverage includes GGUF quantization decoding and concurrent scheduler batching/order behavior.

Key files:

- `src/webgpu/quant-matmul.ts` — Q4_0, Q4_K, and Q6_K fused projection kernels.
- `src/webgpu/ops.ts` — transformer operation kernels.
- `src/webgpu/model.ts` — full 28-layer execution plans.
- `src/webgpu/embedding-engine.ts` — concurrent request micro-batching API.
- `src/gguf/` — GGUF metadata and tensor parsing.
- `scripts/workloads.ts` — reproducible benchmark inputs.

## License

The source code is available under the [MIT License](LICENSE). Qwen model files are not distributed by this repository and retain their upstream Apache-2.0 license.
