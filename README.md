# Qwen3 Embedding 0.6B on WebGPU

A complete, browser-native Qwen3 Embedding 0.6B runtime built from custom WGSL kernels. It loads a GGUF model directly, runs all 28 transformer layers on WebGPU, and micro-batches up to 16 simultaneous embedding requests.

The implementation does **not** delegate inference to ONNX Runtime. Quantized projection, embedding lookup, RMSNorm/residual, Q/K normalization and RoPE, causal attention, SwiGLU, and last-token L2 normalization are implemented in this repository.

![WebGPU and LM Studio benchmark comparison](docs/lm-studio-comparison.svg)

## Required model

The app uses this exact release asset by default:

| Property | Value |
|---|---|
| Release asset | [`qwen3-embedding-0.6b-q4_0-webgpu.gguf`](https://github.com/shihanqu/qwen3-embedding-webgpu/releases/download/model-q4_0-v1/qwen3-embedding-0.6b-q4_0-webgpu.gguf) |
| Quantization | GGUF Q4_0; token embeddings remain Q6_K, as selected by llama.cpp |
| Source | [Qwen/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF), file `Qwen3-Embedding-0.6B-f16.gguf` |
| Generated size | 381,335,744 bytes (364 MiB) |
| SHA-256 | `4acbfc4947344ca4d4a215ee35e601c5e6f505172b517da194460e2ff113433e` |

The 364 MiB weight file is attached to the [`model-q4_0-v1` GitHub Release](https://github.com/shihanqu/qwen3-embedding-webgpu/releases/tag/model-q4_0-v1), rather than stored in normal Git history. Its [artifact notice](MODEL_NOTICE.md), [Apache-2.0 model license](MODEL_LICENSE), and [checksum](docs/model-q4_0-v1.sha256) are checked into the repository and attached to the release.

## Quick start

### 1. Prerequisites

- Node.js 24 or newer and npm.
- Chrome, Edge, or another Chromium browser with WebGPU, `shader-f16`, and subgroup support enabled.

### 2. Install and run

```sh
git clone https://github.com/shihanqu/qwen3-embedding-webgpu.git
cd qwen3-embedding-webgpu
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/?scheduler=1` and select **Load model & benchmark kernel**. The first load downloads the 364 MiB model from the GitHub Release, uploads its weights, downloads Qwen's tokenizer through Transformers.js, and compiles the WebGPU pipelines. Later executions can use the browser cache and reuse compiled plans.

LM Studio is only required for comparison benchmarks. The WebGPU model itself runs entirely in the browser after its model and tokenizer have loaded.

### Local or offline model copy

Download and checksum-verify the release asset into `models/`:

```sh
npm run model:download:webgpu
VITE_Q40_MODEL_URL=/models/qwen3-embedding-0.6b-q4_0-webgpu.gguf npm run dev
```

`VITE_Q40_MODEL_URL` can point to any CORS-accessible mirror or same-origin path. The optional `model:download:q4km` script downloads a higher-quality Q4_K_M model for experiments; launch the app with `?q4km=1` to use it. That alternate model is not used for the published benchmark numbers.

### Reproduce the release artifact

To regenerate the file, build [`llama-quantize`](https://github.com/ggml-org/llama.cpp) from a recent llama.cpp checkout, then quantize directly from Qwen's official f16 model. Do not re-quantize an existing Q4 model: double quantization materially reduces embedding agreement.

```sh
LLAMA_QUANTIZE=/absolute/path/to/llama-quantize \
  npm run model:prepare:q40

shasum -a 256 models/qwen3-embedding-0.6b-q4_0-webgpu.gguf
```

The preparation script downloads the official 1.2 GB f16 GGUF if it is absent, then writes the renamed WebGPU artifact under `models/`. You can instead set `QWEN3_F16_PATH=/path/to/model.gguf`.

## Request API

`Qwen3EmbeddingEngine.embed()` is the request-facing API. Calls made during the same scheduling window are padded after EOS, combined into a batch of at most 16, executed with a cached plan, and resolved in their original order.

```ts
import { AutoTokenizer } from '@huggingface/transformers';
import { GGUFReader, QWEN3_METADATA_KEYS } from './src/gguf/reader.ts';
import { requestWebGPUDevice } from './src/webgpu/device.ts';
import { Qwen3EmbeddingEngine, type TokenizerLike } from './src/webgpu/embedding-engine.ts';
import { Qwen3WebGPUModel } from './src/webgpu/model.ts';

const { device } = await requestWebGPUDevice();
const modelUrl = 'https://github.com/shihanqu/qwen3-embedding-webgpu/releases/download/model-q4_0-v1/qwen3-embedding-0.6b-q4_0-webgpu.gguf';
const bytes = await fetch(modelUrl).then((response) => response.arrayBuffer());
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

- `?scheduler=1` — end-to-end scheduler benchmark through 16 simultaneous `embed()` calls.
- `?sweep=1` — warmed 1/2/4/8/16 sweep at 6, 17, 26, and 105 exact tokenizer tokens.
- `?profile=16` — per-stage GPU timestamps for a 16-request batch.
- `/` — full correctness check and the 105-token stress case.

The default correctness page can optionally compare against llama.cpp running the exact same Q4_0 file:

```sh
llama-server \
  -m models/qwen3-embedding-0.6b-q4_0-webgpu.gguf \
  --embedding --pooling last -ngl 99 \
  --host 127.0.0.1 --port 1236
```

Set `Q4_0_REFERENCE_URL` when that server uses a different address.

## Published benchmark

WebGPU measurements were collected July 14, 2026. The matching LM Studio concurrency runs were collected July 15, 2026 on the same machine:

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

| Exact tokens | WebGPU single | LM Studio single | WebGPU / LM single | WebGPU at 16 concurrent | LM Studio at 16 concurrent | WebGPU / LM at 16 |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 57.33 req/s | 32.22 req/s | **1.78×** | 461.03 req/s | 31.12 req/s | **14.82×** |
| 17 | 51.06 req/s | 28.48 req/s | 1.79× | 186.22 req/s | 30.51 req/s | 6.10× |
| 26 | 49.44 req/s | 29.13 req/s | 1.70× | 130.54 req/s | 31.21 req/s | 4.18× |
| 105 | 25.65 req/s | 20.55 req/s | 1.25× | 38.41 req/s | 21.66 req/s | 1.77× |

LM Studio was tested with 16 independent HTTP workers against the same text used for each WebGPU row. After 10 warmup requests, each condition ran for 10 seconds; the 6-token condition ran for 15 seconds. There were no request errors. LM Studio's aggregate throughput at concurrency 16 remained close to its single-request throughput, while per-request latency rose to roughly 0.5–0.7 seconds.

The WebGPU output agrees with llama.cpp running the exact same clean Q4_0 GGUF at cosine `0.999995`. Batch output agrees with the single path at cosine `1.000000` for the 6-token case. Cosine against a separately quantized LM Studio model is not an implementation correctness comparison.

Raw chart data is checked in at [`docs/benchmarks/2026-07-14-m3-max.json`](docs/benchmarks/2026-07-14-m3-max.json). Regenerate the README graph with `npm run bench:chart`.

## Benchmark methodology

- Record exact Qwen tokenizer length, including EOS; do not rely on word-count estimates.
- Separate cold model download, GPU upload, and pipeline compilation from warmed inference.
- Use identical text and sequential requests for the LM Studio single-stream baseline.
- For LM Studio concurrency, issue independent HTTP requests from 16 workers rather than sending a single array-valued request.
- Sweep concurrency 1, 2, 4, 8, and 16 instead of reporting only the best batch.
- Report both batch latency and aggregate requests per second.
- Compare each batch embedding against the single-request output.
- Keep sequence-length results separate because GPU occupancy and attention cost change substantially with length.

Reproduce the 6-token LM Studio run with:

```sh
npm run bench:baseline -- \
  --workload=tiny --concurrency=1,16 \
  --duration-ms=15000 --warmup=10
```

## Deployment notes

- The production build downloads the pinned `model-q4_0-v1` GitHub Release asset by default. Set `VITE_Q40_MODEL_URL` at build time to use a same-origin copy or another CORS-accessible mirror.
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

The source code is available under the [MIT License](LICENSE). The released Qwen model artifact retains its upstream Apache-2.0 terms; see the [model license](MODEL_LICENSE) and [artifact notice](MODEL_NOTICE.md).
