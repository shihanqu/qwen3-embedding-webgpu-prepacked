import type { GGUFModel, GGUFTensorInfo } from '../gguf/types.ts';
import { GGMLType } from '../gguf/types.ts';
import { createBufferWithData } from './device.ts';
import { PREPACKED_STORAGE_COMPACT, type PrepackedModel } from '../prepacked/format.ts';
import { QuantMatmulKernel, type QuantMatmulRun } from './quant-matmul.ts';
import {
  CausalAttentionKernel,
  EmbeddingLookupKernel,
  LastTokenPoolKernel,
  QKNormRopeKernel,
  RmsNormKernel,
  SwiGLUKernel,
  encodeRun,
  type AttentionRun,
  type EncodableRun,
} from './ops.ts';

const HIDDEN = 1024;
const INTERMEDIATE = 3072;
const Q_WIDTH = 2048;
const LAYERS = 28;
const EPSILON = 1e-6;
const ROPE_THETA = 1_000_000;

interface LayerRuns {
  qkv: QuantMatmulRun;
  qkRope: EncodableRun;
  attention: AttentionRun;
  attentionOutput: QuantMatmulRun;
  postAttentionNorm: EncodableRun;
  gateUp: QuantMatmulRun;
  swiglu: EncodableRun;
  down: QuantMatmulRun;
  postFfnNorm: EncodableRun;
}

function activationBuffer(device: GPUDevice, elements: number, label: string, f32 = false): GPUBuffer {
  return device.createBuffer({
    size: Math.ceil(elements * (f32 ? 4 : 2) / 4) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label,
  });
}

function isPrepackedModel(model: GGUFModel | PrepackedModel): model is PrepackedModel {
  return 'header' in model;
}

export class Qwen3ExecutionPlan {
  readonly ownedBuffers: GPUBuffer[] = [];
  readonly tokenIds: GPUBuffer;
  readonly lengths: GPUBuffer;
  readonly embeddings: GPUBuffer;
  readonly readback: GPUBuffer;
  readonly initialEmbedding: EncodableRun;
  readonly initialNorm: EncodableRun;
  readonly layers: LayerRuns[] = [];
  readonly pool: EncodableRun;

  constructor(readonly model: Qwen3WebGPUModel, readonly batch: number, readonly sequence: number) {
    const { device } = model;
    const tokens = batch * sequence;
    const activation=(elements:number,label:string,f32=false)=>{const buffer=activationBuffer(device,elements,label,f32);this.ownedBuffers.push(buffer);return buffer;};
    this.tokenIds = activation(tokens * 2, 'token ids'); // u32: f16 helper sizing => tokens*4 bytes
    this.lengths = activation(batch * 2, 'sequence lengths');
    const x = activation(tokens * HIDDEN, 'residual stream');
    const noResidual = activation(tokens * HIDDEN, 'unused residual');
    const normalized = activation(tokens * HIDDEN, 'normalized hidden state');
    const qkvRaw = activation(tokens * (Q_WIDTH + HIDDEN * 2), 'query + key + value raw');
    const q = activation(tokens * Q_WIDTH, 'query rope');
    const k = activation(tokens * HIDDEN, 'key rope');
    const attention = activation(tokens * Q_WIDTH, 'attention');
    const delta = activation(tokens * HIDDEN, 'residual delta');
    const gateUp = activation(tokens * INTERMEDIATE * 2, 'FFN gate + up');
    const ffn = activation(tokens * INTERMEDIATE, 'SwiGLU output');
    const ropeValues = new Float32Array(sequence * 64 * 2);
    for (let position = 0; position < sequence; position += 1) for (let dimension = 0; dimension < 64; dimension += 1) {
      const angle = position * Math.pow(ROPE_THETA, -dimension / 64);
      const offset = (position * 64 + dimension) * 2;
      ropeValues[offset] = Math.cos(angle); ropeValues[offset + 1] = Math.sin(angle);
    }
    const ropeTable = createBufferWithData(device, ropeValues, GPUBufferUsage.STORAGE, 'RoPE sin/cos table');
    this.ownedBuffers.push(ropeTable);
    this.embeddings = activation(batch * HIDDEN, 'embeddings', true);
    this.readback = device.createBuffer({ size: batch * HIDDEN * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, label: 'embedding readback' });
    this.ownedBuffers.push(this.readback);

    this.initialEmbedding = model.embedding.createRun(this.tokenIds, model.weight('token_embd.weight'), x, tokens);
    this.initialNorm = model.rmsNorm.createRun(x, noResidual, model.weight('blk.0.attn_norm.weight'), normalized, tokens, HIDDEN, EPSILON, false);

    for (let layer = 0; layer < LAYERS; layer += 1) {
      const prefix = `blk.${layer}`;
      const qkvRun = model.matmul(normalized, `${prefix}.attn_qkv.weight`, tokens, qkvRaw);
      const qkRope = model.qkNormRope.createRun(qkvRaw, model.weight(`${prefix}.attn_q_norm.weight`), model.weight(`${prefix}.attn_k_norm.weight`), q, k, tokens, sequence, EPSILON, ropeTable, 4096);
      const attentionRun = model.attention.createRun(q, k, qkvRaw, attention, batch, sequence, 16, 8, 4096, 3072);
      const attentionOutput = model.matmul(attention, `${prefix}.attn_output.weight`, tokens, delta);
      const postAttentionNorm = model.rmsNorm.createRun(x, delta, model.weight(`${prefix}.ffn_norm.weight`), normalized, tokens, HIDDEN, EPSILON, true);
      const gateUpRun = model.matmul(normalized, `${prefix}.ffn_gate_up.weight`, tokens, gateUp);
      const swiglu = model.swiglu.createRun(gateUp, ffn, tokens, INTERMEDIATE);
      const nextNormName = layer + 1 < LAYERS ? `blk.${layer + 1}.attn_norm.weight` : 'output_norm.weight';
      const downRun = model.matmul(ffn, `${prefix}.ffn_down.weight`, tokens, delta);
      const postFfnNorm = model.rmsNorm.createRun(x, delta, model.weight(nextNormName), normalized, tokens, HIDDEN, EPSILON, true);
      this.layers.push({ qkv: qkvRun, qkRope, attention: attentionRun, attentionOutput, postAttentionNorm, gateUp: gateUpRun, swiglu, down: downRun, postFfnNorm });
    }
    this.pool = model.pool.createRun(normalized, this.lengths, this.embeddings, batch, sequence);
  }

  destroy():void{
    for(const layer of this.layers)for(const buffer of layer.attention.ownedBuffers)buffer.destroy();
    for(const buffer of this.ownedBuffers)buffer.destroy();
  }

  async run(ids: Uint32Array, sequenceLengths: Uint32Array): Promise<Float32Array[]> {
    if (ids.length !== this.batch * this.sequence) throw new Error('token buffer does not match execution plan');
    this.model.device.queue.writeBuffer(this.tokenIds, 0, ids.buffer as ArrayBuffer, ids.byteOffset, ids.byteLength);
    this.model.device.queue.writeBuffer(this.lengths, 0, sequenceLengths.buffer as ArrayBuffer, sequenceLengths.byteOffset, sequenceLengths.byteLength);
    const encoder = this.model.device.createCommandEncoder({ label: `Qwen3 batch=${this.batch} sequence=${this.sequence}` });
    const pass = encoder.beginComputePass();
    encodeRun(pass, this.initialEmbedding);
    encodeRun(pass, this.initialNorm);
    for (const layer of this.layers) {
      this.model.kernelFor(layer.qkv).encode(pass, layer.qkv);
      encodeRun(pass, layer.qkRope); this.model.attention.encode(pass, layer.attention);
      this.model.kernelFor(layer.attentionOutput).encode(pass, layer.attentionOutput);
      encodeRun(pass, layer.postAttentionNorm);
      this.model.kernelFor(layer.gateUp).encode(pass, layer.gateUp);
      encodeRun(pass, layer.swiglu);
      this.model.kernelFor(layer.down).encode(pass, layer.down);
      encodeRun(pass, layer.postFfnNorm);
    }
    encodeRun(pass, this.pool);
    pass.end();
    encoder.copyBufferToBuffer(this.embeddings, 0, this.readback, 0, this.batch * HIDDEN * 4);
    this.model.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const flat = new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    return Array.from({ length: this.batch }, (_, index) => flat.slice(index * HIDDEN, (index + 1) * HIDDEN));
  }

  async profile(ids: Uint32Array, sequenceLengths: Uint32Array): Promise<Record<string, number>> {
    if (!this.model.device.features.has('timestamp-query')) throw new Error('timestamp-query is unavailable');
    this.model.device.queue.writeBuffer(this.tokenIds, 0, ids.buffer as ArrayBuffer, ids.byteOffset, ids.byteLength);
    this.model.device.queue.writeBuffer(this.lengths, 0, sequenceLengths.buffer as ArrayBuffer, sequenceLengths.byteOffset, sequenceLengths.byteLength);
    const stageCount = 2 + this.layers.length * 8;
    const querySet = this.model.device.createQuerySet({ type: 'timestamp', count: stageCount * 2 });
    const queryBytes = stageCount * 16;
    const resolve = this.model.device.createBuffer({ size: queryBytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const readback = this.model.device.createBuffer({ size: queryBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.model.device.createCommandEncoder({ label: 'profiled Qwen3 graph' });
    const names: string[] = [];
    let query = 0;
    const stage = (name: string, encode: (pass: GPUComputePassEncoder) => void) => {
      names.push(name);
      const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: query++, endOfPassWriteIndex: query++ } });
      encode(pass); pass.end();
    };
    stage('embedding_norm', (pass) => { encodeRun(pass, this.initialEmbedding); encodeRun(pass, this.initialNorm); });
    for (const layer of this.layers) {
      stage('qkv', (pass) => this.model.kernelFor(layer.qkv).encode(pass, layer.qkv));
      stage('rope_attention', (pass) => { encodeRun(pass, layer.qkRope); this.model.attention.encode(pass, layer.attention); });
      stage('attention_output', (pass) => this.model.kernelFor(layer.attentionOutput).encode(pass, layer.attentionOutput));
      stage('post_attention_norm', (pass) => encodeRun(pass, layer.postAttentionNorm));
      stage('ffn_up_gate', (pass) => this.model.kernelFor(layer.gateUp).encode(pass, layer.gateUp));
      stage('swiglu', (pass) => encodeRun(pass, layer.swiglu));
      stage('ffn_down', (pass) => this.model.kernelFor(layer.down).encode(pass, layer.down));
      stage('post_ffn_norm', (pass) => encodeRun(pass, layer.postFfnNorm));
    }
    stage('pool', (pass) => encodeRun(pass, this.pool));
    encoder.resolveQuerySet(querySet, 0, query, resolve, 0);
    encoder.copyBufferToBuffer(resolve, 0, readback, 0, queryBytes);
    this.model.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const timestamps = new BigUint64Array(readback.getMappedRange().slice(0));
    const totals: Record<string, number> = {};
    for (let index = 0; index < names.length; index += 1) {
      const milliseconds = Number(timestamps[index * 2 + 1] - timestamps[index * 2]) / 1e6;
      totals[names[index]] = (totals[names[index]] ?? 0) + milliseconds;
    }
    readback.destroy(); resolve.destroy(); querySet.destroy();
    return totals;
  }
}

export class Qwen3WebGPUModel {
  readonly weights = new Map<string, GPUBuffer>();
  readonly tensorInfo = new Map<string, GGUFTensorInfo>();
  readonly q4: QuantMatmulKernel;
  readonly q40: QuantMatmulKernel;
  readonly q40Prepacked: QuantMatmulKernel;
  readonly q6: QuantMatmulKernel;
  readonly rmsNorm: RmsNormKernel;
  readonly swiglu: SwiGLUKernel;
  readonly qkNormRope: QKNormRopeKernel;
  readonly attention: CausalAttentionKernel;
  readonly pool: LastTokenPoolKernel;
  readonly embedding: EmbeddingLookupKernel;
  readonly prepackedNames = new Set<string>();

  constructor(readonly device: GPUDevice, source: GGUFModel | PrepackedModel) {
    this.q4 = new QuantMatmulKernel(device, GGMLType.Q4_K);
    this.q40 = new QuantMatmulKernel(device, GGMLType.Q4_0);
    this.q40Prepacked = new QuantMatmulKernel(device, GGMLType.Q4_0, 'q4_0-tile32-compact');
    this.q6 = new QuantMatmulKernel(device, GGMLType.Q6_K);
    this.rmsNorm = new RmsNormKernel(device); this.swiglu = new SwiGLUKernel(device);
    this.qkNormRope = new QKNormRopeKernel(device); this.attention = new CausalAttentionKernel(device);
    this.pool = new LastTokenPoolKernel(device); this.embedding = new EmbeddingLookupKernel(device);
    if (isPrepackedModel(source)) {
      for (const tensor of source.tensors.values()) {
        const bytes = new Uint8Array(source.buffer, tensor.byteOffset, tensor.byteLength);
        this.weights.set(tensor.name, createBufferWithData(device, bytes, GPUBufferUsage.STORAGE, tensor.name));
        this.tensorInfo.set(tensor.name, {
          name: tensor.name,
          dimensions: tensor.dimensions,
          type: tensor.type,
          offset: tensor.offset,
          byteOffset: tensor.byteOffset,
          byteLength: tensor.byteLength,
          elementCount: tensor.dimensions.reduce((product, value) => product * value, 1),
        });
        if (tensor.storage === PREPACKED_STORAGE_COMPACT) this.prepackedNames.add(tensor.name);
      }
      if (this.prepackedNames.size !== LAYERS * 4) throw new Error(`prepacked model has ${this.prepackedNames.size} compact matrices; expected ${LAYERS * 4}`);
      return;
    }

    const gguf = source;
    for (const tensor of gguf.tensors.values()) {
      const bytes = new Uint8Array(gguf.buffer, tensor.byteOffset, tensor.byteLength);
      this.weights.set(tensor.name, createBufferWithData(device, bytes, GPUBufferUsage.STORAGE, tensor.name));
      this.tensorInfo.set(tensor.name, tensor);
    }
    const combineQ4 = (name: string, ...sourceNames: string[]) => {
      const sources = sourceNames.map((sourceName) => gguf.tensors.get(sourceName));
      const first = sources[0];
      if (!first || sources.some((tensor) => !tensor || tensor.type !== first.type || (tensor.type !== GGMLType.Q4_K && tensor.type !== GGMLType.Q4_0) || tensor.dimensions[0] !== first.dimensions[0])) {
        throw new Error(`cannot combine ${sourceNames.join(', ')}`);
      }
      const tensors = sources as GGUFTensorInfo[];
      const bytes = new Uint8Array(tensors.reduce((sum, tensor) => sum + tensor.byteLength, 0));
      let byteOffset = 0;
      for (const tensor of tensors) { bytes.set(new Uint8Array(gguf.buffer, tensor.byteOffset, tensor.byteLength), byteOffset); byteOffset += tensor.byteLength; }
      this.weights.set(name, createBufferWithData(device, bytes, GPUBufferUsage.STORAGE, name));
      this.tensorInfo.set(name, { name, dimensions: [first.dimensions[0], tensors.reduce((sum, tensor) => sum + tensor.dimensions[1], 0)], type: first.type, offset: 0, byteOffset: 0, byteLength: bytes.byteLength, elementCount: tensors.reduce((sum, tensor) => sum + tensor.elementCount, 0) });
    };
    for (let layer = 0; layer < LAYERS; layer += 1) {
      combineQ4(`blk.${layer}.attn_qkv.weight`, `blk.${layer}.attn_q.weight`, `blk.${layer}.attn_k.weight`, `blk.${layer}.attn_v.weight`);
      combineQ4(`blk.${layer}.ffn_gate_up.weight`, `blk.${layer}.ffn_gate.weight`, `blk.${layer}.ffn_up.weight`);
    }
  }

  weight(name: string): GPUBuffer {
    const value = this.weights.get(name);
    if (!value) throw new Error(`missing tensor ${name}`);
    return value;
  }

  matmul(input: GPUBuffer, weightName: string, rows: number, output: GPUBuffer): QuantMatmulRun {
    const info = this.tensorInfo.get(weightName);
    if (!info || info.dimensions.length !== 2) throw new Error(`invalid matrix ${weightName}`);
    const [k, n] = info.dimensions;
    const kernel = this.prepackedNames.has(weightName) ? this.q40Prepacked : info.type===GGMLType.Q4_0?this.q40:info.type === GGMLType.Q4_K ? this.q4 : info.type === GGMLType.Q6_K ? this.q6 : undefined;
    if (!kernel) throw new Error(`unsupported matrix type ${info.type} for ${weightName}`);
    return kernel.createRunFromBuffers(input, this.weight(weightName), rows, n, k, output);
  }

  kernelFor(run: QuantMatmulRun): QuantMatmulKernel {
    if(run.pipeline===this.q40Prepacked.latencyPipeline||run.pipeline===this.q40Prepacked.throughputPipeline||run.pipeline===this.q40Prepacked.midBatchPipeline||run.pipeline===this.q40Prepacked.batchPipeline||run.pipeline===this.q40Prepacked.subgroupPipeline)return this.q40Prepacked;
    if(run.pipeline===this.q40.latencyPipeline||run.pipeline===this.q40.throughputPipeline||run.pipeline===this.q40.midBatchPipeline||run.pipeline===this.q40.batchPipeline)return this.q40;
    return run.pipeline === this.q4.latencyPipeline || run.pipeline === this.q4.throughputPipeline || run.pipeline === this.q4.midBatchPipeline || run.pipeline === this.q4.batchPipeline ? this.q4 : this.q6;
  }

  createPlan(batch: number, sequence: number): Qwen3ExecutionPlan { return new Qwen3ExecutionPlan(this, batch, sequence); }
}
