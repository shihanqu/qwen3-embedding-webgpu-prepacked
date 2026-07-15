import { GGUFReader, QWEN3_METADATA_KEYS } from './gguf/reader.ts';
import { dequantizeQ4KBlock, dequantizeQ6KBlock, halfToFloat } from './gguf/quantization.ts';
import { GGMLType } from './gguf/types.ts';
import { toFloat16Bits } from './math/f16.ts';
import { requestWebGPUDevice } from './webgpu/device.ts';
import { Qwen3EmbeddingEngine, type TokenizerLike } from './webgpu/embedding-engine.ts';
import { QuantMatmulKernel } from './webgpu/quant-matmul.ts';
import { CausalAttentionKernel, LastTokenPoolKernel, QKNormRopeKernel, RmsNormKernel, SwiGLUKernel } from './webgpu/ops.ts';
import { Qwen3WebGPUModel } from './webgpu/model.ts';
import { getWorkload } from '../scripts/workloads.ts';

const runButton = document.querySelector<HTMLButtonElement>('#run')!;
const output = document.querySelector<HTMLElement>('#output')!;
const DEFAULT_WEBGPU_MODEL_URL = '/model-release/qwen3-embedding-0.6b-q4_0-webgpu.gguf';

function write(line: string): void {
  output.textContent += `\n${line}`;
}

async function readF16Buffer(device: GPUDevice, source: GPUBuffer, values: number): Promise<Uint16Array> {
  const bytes = Math.ceil(values * 2 / 4) * 4;
  const readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const result = new Uint16Array(readback.getMappedRange().slice(0));
  readback.destroy();
  return result;
}

async function runOrtBenchmark(): Promise<void> {
  output.textContent = 'Loading ONNX Q4F16 WebGPU reference…';
  const { pipeline } = await import('@huggingface/transformers');
  const extractor = await pipeline('feature-extraction', 'onnx-community/Qwen3-Embedding-0.6B-ONNX', {
    dtype: 'q4f16',
    device: 'webgpu',
  });
  const text = getWorkload('acceptance').inputs[0];
  const options = { pooling: 'last_token', normalize: true } as const;
  await extractor(text, options);
  const singleRepeats = 3;
  const singleStarted = performance.now();
  for (let index = 0; index < singleRepeats; index += 1) await extractor(text, options);
  const singleMs = (performance.now() - singleStarted) / singleRepeats;
  write(`ORT Q4F16 single: ${(1000 / singleMs).toFixed(2)} req/s (${singleMs.toFixed(1)} ms)`);
  const batch = Array(16).fill(text);
  await extractor(batch, options);
  const batchStarted = performance.now();
  await extractor(batch, options);
  const batchMs = performance.now() - batchStarted;
  const aggregate = 16_000 / batchMs;
  write(`ORT Q4F16 batch 16: ${aggregate.toFixed(2)} req/s (${batchMs.toFixed(1)} ms), scaling ${(aggregate / (1000 / singleMs)).toFixed(2)}×`);
  write('PASS');
}

runButton.addEventListener('click', async () => {
  runButton.disabled = true;
  output.textContent = 'Requesting WebGPU adapter…';
  try {
    if (new URLSearchParams(location.search).has('ort')) {
      await runOrtBenchmark();
      return;
    }
    const { adapter, device } = await requestWebGPUDevice();
    write(`Adapter: ${adapter.info?.description || adapter.info?.device || 'WebGPU GPU'}`);
    write(`Features: ${Array.from(adapter.features).sort().join(', ')}`);
    write(`Storage binding limit: ${(device.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB`);
    const searchParams = new URLSearchParams(location.search);
    const q40 = !searchParams.has('q4km');
    const modelUrl = q40
      ? (import.meta.env.VITE_Q40_MODEL_URL?.trim() || DEFAULT_WEBGPU_MODEL_URL)
      : '/models/qwen3-embedding-0.6b-q4_k_m.gguf';
    write(q40 ? 'Fetching 364 MiB Q4_0 WebGPU release model…' : 'Fetching 378 MiB Q4_K_M model…');
    const response = await fetch(modelUrl);
    if (!response.ok) throw new Error(`model fetch failed: ${response.status}`);
    const buffer = await response.arrayBuffer();
    const model = new GGUFReader(buffer).parse({ metadataKeys: QWEN3_METADATA_KEYS });
    write(`Parsed ${model.metadata.get('general.name')}: ${model.tensors.size} tensors`);
    const acceptanceOnly = searchParams.has('acceptance') || searchParams.has('profile') || searchParams.has('sweep') || searchParams.has('scheduler');
    const skipMicrobenchmarks = acceptanceOnly || q40;

    if (!skipMicrobenchmarks) {
    const tensor = model.tensors.get('blk.0.ffn_gate.weight');
    if (!tensor || tensor.type !== GGMLType.Q4_K) throw new Error('expected Q4_K gate tensor');
    const [k, n] = tensor.dimensions;
    const weights = new Uint8Array(buffer, tensor.byteOffset, tensor.byteLength);
    device.pushErrorScope('validation');
    const kernel = new QuantMatmulKernel(device, GGMLType.Q4_K);
    const pipelineError = await device.popErrorScope();
    if (pipelineError) throw new Error(`Q4_K pipeline validation: ${pipelineError.message}`);

    for (const m of [128, 2048]) {
      const inputValues = Float32Array.from({ length: m * k }, (_, index) => ((index % 29) - 14) / 29);
      const run = kernel.createRun(toFloat16Bits(inputValues), weights, m, n, k);
      kernel.dispatch(run, 2);
      await device.queue.onSubmittedWorkDone();
      const repetitions = m === 128 ? 20 : 5;
      const started = performance.now();
      kernel.dispatch(run, repetitions);
      await device.queue.onSubmittedWorkDone();
      const elapsed = performance.now() - started;
      const milliseconds = elapsed / repetitions;
      const tflops = (2 * m * n * k) / (milliseconds / 1000) / 1e12;
      write(`Q4_K ${m}×${k} · ${k}×${n}: ${milliseconds.toFixed(2)} ms, ${tflops.toFixed(2)} TFLOP/s`);

      if (m === 128) {
        const resultBits = await readF16Buffer(device, run.output, m * n);
        const firstWeightRow = new Float32Array(k);
        for (let block = 0; block < k / 256; block += 1) {
          firstWeightRow.set(dequantizeQ4KBlock(weights, block * 144), block * 256);
        }
        let expected = 0;
        for (let index = 0; index < k; index += 1) expected += inputValues[index] * firstWeightRow[index];
        const actual = halfToFloat(resultBits[0]);
        const error = Math.abs(actual - expected);
        write(`CPU/GPU check: expected ${expected.toFixed(4)}, got ${actual.toFixed(4)}, abs error ${error.toFixed(4)}`);
        if (error > Math.max(0.01, Math.abs(expected) * 0.02)) throw new Error('Q4_K GPU result exceeds tolerance');
      }
    }

    const q6Tensor = model.tensors.get('blk.0.attn_v.weight');
    if (!q6Tensor || q6Tensor.type !== GGMLType.Q6_K) throw new Error('expected Q6_K value tensor');
    const [q6K, q6N] = q6Tensor.dimensions;
    const q6Weights = new Uint8Array(buffer, q6Tensor.byteOffset, q6Tensor.byteLength);
    device.pushErrorScope('validation');
    const q6Kernel = new QuantMatmulKernel(device, GGMLType.Q6_K);
    const q6PipelineError = await device.popErrorScope();
    if (q6PipelineError) throw new Error(`Q6_K pipeline validation: ${q6PipelineError.message}`);
    for (const m of [128, 2048]) {
      const inputValues = Float32Array.from({ length: m * q6K }, (_, index) => ((index % 29) - 14) / 29);
      const q6Run = q6Kernel.createRun(toFloat16Bits(inputValues), q6Weights, m, q6N, q6K);
      q6Kernel.dispatch(q6Run, 2);
      await device.queue.onSubmittedWorkDone();
      const repetitions = m === 128 ? 20 : 5;
      const started = performance.now();
      q6Kernel.dispatch(q6Run, repetitions);
      await device.queue.onSubmittedWorkDone();
      const milliseconds = (performance.now() - started) / repetitions;
      const tflops = (2 * m * q6N * q6K) / (milliseconds / 1000) / 1e12;
      write(`Q6_K ${m}×${q6K} · ${q6K}×${q6N}: ${milliseconds.toFixed(2)} ms, ${tflops.toFixed(2)} TFLOP/s`);
      if (m === 128) {
        const bits = await readF16Buffer(device, q6Run.output, m * q6N);
        const firstWeightRow = new Float32Array(q6K);
        for (let block = 0; block < q6K / 256; block += 1) {
          firstWeightRow.set(dequantizeQ6KBlock(q6Weights, block * 210), block * 256);
        }
        let expected = 0;
        for (let index = 0; index < q6K; index += 1) expected += inputValues[index] * firstWeightRow[index];
        const actual = halfToFloat(bits[0]);
        const error = Math.abs(actual - expected);
        write(`Q6 CPU/GPU check: expected ${expected.toFixed(4)}, got ${actual.toFixed(4)}, abs error ${error.toFixed(4)}`);
        if (error > Math.max(0.01, Math.abs(expected) * 0.02)) throw new Error('Q6_K GPU result exceeds tolerance');
      }
    }
    device.pushErrorScope('validation');
    new RmsNormKernel(device);
    new SwiGLUKernel(device);
    new QKNormRopeKernel(device);
    new CausalAttentionKernel(device);
    new LastTokenPoolKernel(device);
    const opPipelineError = await device.popErrorScope();
    if (opPipelineError) throw new Error(`transformer pipeline validation: ${opPipelineError.message}`);
    write('Transformer fusion pipelines compiled');
    if (new URLSearchParams(location.search).has('kernel')) { write('PASS'); return; }
    }

    write('Uploading all model tensors to the GPU…');
    device.pushErrorScope('validation');
    const runtime = new Qwen3WebGPUModel(device, model);
    const uploadError = await device.popErrorScope();
    if (uploadError) throw new Error(`model upload validation: ${uploadError.message}`);
    write('Loading tokenizer…');
    const { AutoTokenizer } = await import('@huggingface/transformers');
    const tokenizer = await AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B');
    if (searchParams.has('scheduler')) {
      const engine = new Qwen3EmbeddingEngine(runtime, tokenizer as unknown as TokenizerLike, 16, 0);
      // Six tokenizer tokens including EOS: a stable latency/concurrency acceptance case.
      // The broader sweep below separately exposes scaling at 17, 26, and 105 tokens.
      const schedulerText = 'WebGPU embedding benchmark.';
      const reference = await engine.embed(schedulerText);
      const singleRepeats = 5;
      const singleStarted = performance.now();
      for (let index = 0; index < singleRepeats; index += 1) await engine.embed(schedulerText);
      const singleRps = singleRepeats * 1000 / (performance.now() - singleStarted);
      const concurrentRepeats = 3;
      const concurrentStarted = performance.now();
      let batchResults: Float32Array[] = [];
      for (let repeat = 0; repeat < concurrentRepeats; repeat += 1) {
        batchResults = await Promise.all(Array.from({ length: 16 }, () => engine.embed(schedulerText)));
      }
      const aggregateRps = concurrentRepeats * 16_000 / (performance.now() - concurrentStarted);
      const worstCosine = Math.min(...batchResults.map((embedding) => reference.reduce((sum, value, index) => sum + value * embedding[index], 0)));
      for (let index = 0; index < 2; index += 1) await fetch('/baseline/v1/embeddings', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'text-embedding-qwen3-embedding-0.6b', input: schedulerText }) });
      const baselineStarted = performance.now();
      for (let index = 0; index < 5; index += 1) await fetch('/baseline/v1/embeddings', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'text-embedding-qwen3-embedding-0.6b', input: schedulerText }) });
      const baselineRps = 5000 / (performance.now() - baselineStarted);
      const improvement = singleRps / baselineRps;
      const scaling = aggregateRps / singleRps;
      write(`Scheduler single: ${singleRps.toFixed(2)} req/s; LM Studio ${baselineRps.toFixed(2)} req/s; improvement ${improvement.toFixed(2)}×`);
      write(`Scheduler 16 concurrent: ${aggregateRps.toFixed(2)} aggregate req/s; scaling ${scaling.toFixed(2)}×; worst cosine ${worstCosine.toFixed(6)}`);
      if (worstCosine < 0.999) throw new Error(`scheduler batch cosine ${worstCosine} is below 0.999`);
      write('Benchmark complete');
      return;
    }
    if (!acceptanceOnly) {
    const smokeText = 'The quick brown fox jumps over the lazy dog.';
    const tokenized = tokenizer(smokeText) as unknown as { input_ids: { tolist(): number[][] } };
    const textTokens = tokenized.input_ids.tolist()[0].map(Number);
    if (textTokens[textTokens.length - 1] !== 151643) textTokens.push(151643);
    const sequence = textTokens.length;
    const ids = new Uint32Array(sequence);
    ids.fill(151643);
    ids.set(textTokens);
    write(`Running full 28-layer graph (${textTokens.length} tokens, padded to ${sequence})…`);
    device.pushErrorScope('validation');
    const plan = runtime.createPlan(1, sequence);
    const planError = await device.popErrorScope();
    if (planError) throw new Error(`execution plan validation: ${planError.message}`);
    device.pushErrorScope('validation');
    const fullStarted = performance.now();
    const [embedding] = await plan.run(ids, new Uint32Array([textTokens.length]));
    const fullElapsed = performance.now() - fullStarted;
    const graphError = await device.popErrorScope().catch(() => null);
    if (graphError) throw new Error(`graph validation: ${graphError.message}`);
    const norm = Math.sqrt(embedding.reduce((sum, value) => sum + value * value, 0));
    if (!embedding.every(Number.isFinite) || Math.abs(norm - 1) > 0.01) throw new Error(`invalid embedding norm ${norm}`);
    write(`Full embedding: ${fullElapsed.toFixed(1)} ms, norm ${norm.toFixed(6)}, first [${Array.from(embedding.slice(0, 4), value => value.toFixed(6)).join(', ')}]`);
    const baselineResponse = await fetch('/baseline/v1/embeddings', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'text-embedding-qwen3-embedding-0.6b', input: smokeText }) });
    if (!baselineResponse.ok) throw new Error(`baseline comparison failed: ${baselineResponse.status}`);
    const baselinePayload = await baselineResponse.json() as { data: Array<{ embedding: number[] }> };
    const baselineEmbedding = baselinePayload.data[0].embedding;
    const dot = embedding.reduce((sum, value, index) => sum + value * baselineEmbedding[index], 0);
    const baselineNorm = Math.sqrt(baselineEmbedding.reduce((sum, value) => sum + value * value, 0));
    const cosine = dot / (norm * baselineNorm);
    write(`LM Studio cosine agreement: ${cosine.toFixed(6)}`);
    const baselineCosineFloor = q40 ? 0.90 : 0.95;
    if (cosine < baselineCosineFloor) throw new Error(`embedding cosine agreement ${cosine} is below ${baselineCosineFloor}`);
    const referenceUrl = q40 ? '/q40-reference/v1/embeddings' : '/q4-reference/v1/embeddings';
    const q4Response = await fetch(referenceUrl, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'qwen', input: smokeText }) });
    if (q4Response.ok) {
      const q4Payload = await q4Response.json() as { data: Array<{ embedding: number[] }> };
      const q4Embedding = q4Payload.data[0].embedding;
      const q4Dot = embedding.reduce((sum, value, index) => sum + value * q4Embedding[index], 0);
      const q4Norm = Math.sqrt(q4Embedding.reduce((sum, value) => sum + value * value, 0));
      const q4Cosine = q4Dot / (norm * q4Norm);
      write(`Exact ${q40 ? 'Q4_0' : 'Q4_K_M'} llama.cpp agreement: ${q4Cosine.toFixed(6)}`);
      if (q4Cosine < 0.995) throw new Error(`exact-model cosine agreement ${q4Cosine} is below 0.995`);
    }
    }

    const acceptanceText = getWorkload('acceptance').inputs[0];
    const acceptanceEncoded = tokenizer(acceptanceText) as unknown as { input_ids: { tolist(): number[][] } };
    const acceptanceTokens = acceptanceEncoded.input_ids.tolist()[0].map(Number);
    if (acceptanceTokens[acceptanceTokens.length - 1] !== 151643) acceptanceTokens.push(151643);
    const acceptanceSequence = acceptanceTokens.length;
    const makeBatch = (batch: number) => {
      const batchIds = new Uint32Array(batch * acceptanceSequence); batchIds.fill(151643);
      for (let index = 0; index < batch; index += 1) batchIds.set(acceptanceTokens, index * acceptanceSequence);
      return { ids: batchIds, lengths: new Uint32Array(batch).fill(acceptanceTokens.length) };
    };
    if (searchParams.has('sweep')) {
      write('Concurrency sweep (identical requests, warmed plans):');
      const allSweepWorkloads = [
        ['tiny', 'WebGPU embedding benchmark.'],
        ['sentence', 'A red fox crosses the quiet trail while the morning fog lifts from the valley.'],
        ['short', getWorkload('short').inputs[0]],
        ['acceptance', getWorkload('acceptance').inputs[0]],
      ] as const;
      const sweepWorkloads = searchParams.has('quick') ? allSweepWorkloads.slice(1, 2) : allSweepWorkloads;
      for (const [workloadName, sweepText] of sweepWorkloads) {
        const encoded = tokenizer(sweepText) as unknown as { input_ids: { tolist(): number[][] } };
        const tokens = encoded.input_ids.tolist()[0].map(Number);
        if (tokens[tokens.length - 1] !== 151643) tokens.push(151643);
        let reference: Float32Array | undefined;
        let singleThroughput = 0;
        for (const batchSize of [1, 2, 4, 8, 16]) {
          const ids = new Uint32Array(batchSize * tokens.length);
          for (let request = 0; request < batchSize; request += 1) ids.set(tokens, request * tokens.length);
          const lengths = new Uint32Array(batchSize).fill(tokens.length);
          const plan = runtime.createPlan(batchSize, tokens.length);
          const warm = await plan.run(ids, lengths);
          if (batchSize === 1) reference = warm[0];
          const repeats = batchSize <= 4 ? 3 : 2;
          const started = performance.now();
          let result = warm;
          for (let repeat = 0; repeat < repeats; repeat += 1) result = await plan.run(ids, lengths);
          const batchMs = (performance.now() - started) / repeats;
          const throughput = batchSize * 1000 / batchMs;
          if (batchSize === 1) singleThroughput = throughput;
          const cosine = reference!.reduce((sum, value, index) => sum + value * result[0][index], 0);
          write(`  ${workloadName}/${tokens.length} tokens batch ${batchSize}: ${throughput.toFixed(2)} req/s, ${batchMs.toFixed(1)} ms, ${batchSize === 1 ? '1.00' : (throughput / singleThroughput).toFixed(2)}×, cosine ${cosine.toFixed(6)}`);
        }
        for (let index = 0; index < 2; index += 1) await fetch('/baseline/v1/embeddings', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'text-embedding-qwen3-embedding-0.6b', input: sweepText }) });
        const baselineStarted = performance.now();
        for (let index = 0; index < 5; index += 1) await fetch('/baseline/v1/embeddings', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'text-embedding-qwen3-embedding-0.6b', input: sweepText }) });
        const baselineThroughput = 5000 / (performance.now() - baselineStarted);
        write(`  ${workloadName} LM Studio: ${baselineThroughput.toFixed(2)} req/s; single improvement ${(singleThroughput / baselineThroughput).toFixed(2)}×`);
      }
      write('PASS');
      return;
    }
    write(`Acceptance workload: ${acceptanceTokens.length} exact tokens, bucket ${acceptanceSequence}`);
    const singlePlan = runtime.createPlan(1, acceptanceSequence);
    const singleBatch = makeBatch(1);
    if (searchParams.has('profile')) {
      const profileBatchSize = Math.max(1, Number(searchParams.get('profile')) || 1);
      const profilePlan = profileBatchSize === 1 ? singlePlan : runtime.createPlan(profileBatchSize, acceptanceSequence);
      const profileBatch = makeBatch(profileBatchSize);
      await profilePlan.run(profileBatch.ids, profileBatch.lengths);
      const profile = await profilePlan.profile(profileBatch.ids, profileBatch.lengths);
      write(`GPU profile: ${Object.entries(profile).map(([name, ms]) => `${name}=${ms.toFixed(2)}ms`).join(', ')}`);
      write('PASS');
      return;
    }
    await singlePlan.run(singleBatch.ids, singleBatch.lengths);
    const singleRepeats = 3;
    let singleElapsed = 0;
    for (let index = 0; index < singleRepeats; index += 1) {
      const started = performance.now(); await singlePlan.run(singleBatch.ids, singleBatch.lengths); singleElapsed += performance.now() - started;
    }
    const singleRps = 1000 / (singleElapsed / singleRepeats);
    write(`WebGPU single stream: ${singleRps.toFixed(2)} req/s (${(singleElapsed / singleRepeats).toFixed(1)} ms)`);

    const concurrentPlan = runtime.createPlan(16, acceptanceSequence);
    const concurrentBatch = makeBatch(16);
    await concurrentPlan.run(concurrentBatch.ids, concurrentBatch.lengths);
    const concurrentRepeats = 2;
    let concurrentElapsed = 0;
    for (let index = 0; index < concurrentRepeats; index += 1) {
      const started = performance.now(); await concurrentPlan.run(concurrentBatch.ids, concurrentBatch.lengths); concurrentElapsed += performance.now() - started;
    }
    const aggregateRps = 16_000 / (concurrentElapsed / concurrentRepeats);
    const scaling = aggregateRps / singleRps;
    write(`WebGPU 16-request batch: ${aggregateRps.toFixed(2)} aggregate req/s (${(concurrentElapsed / concurrentRepeats).toFixed(1)} ms/batch), scaling ${scaling.toFixed(2)}×`);

    for (let index = 0; index < 2; index += 1) await fetch('/baseline/v1/embeddings', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'text-embedding-qwen3-embedding-0.6b', input: acceptanceText }) });
    const baselineRepeats = 5;
    const baselineStarted = performance.now();
    for (let index = 0; index < baselineRepeats; index += 1) await fetch('/baseline/v1/embeddings', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'text-embedding-qwen3-embedding-0.6b', input: acceptanceText }) });
    const baselineRps = baselineRepeats * 1000 / (performance.now() - baselineStarted);
    const improvement = singleRps / baselineRps;
    write(`LM Studio same-text baseline: ${baselineRps.toFixed(2)} req/s; WebGPU improvement ${improvement.toFixed(2)}×`);
    write('Benchmark complete');
  } catch (error) {
    console.error(error);
    write(`ERROR: ${error instanceof Error ? error.message : String(error)}`);
  } finally {
    runButton.disabled = false;
  }
});
