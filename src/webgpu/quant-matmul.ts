import { GGMLType } from '../gguf/types.ts';
import { createBufferWithData } from './device.ts';

const TILE_M = 16;
const TILE_N = 16;
const TILE_K = 256;

const shaderPrelude = /* wgsl */`
enable f16;

struct Params {
  m: u32,
  n: u32,
  k: u32,
  blocks_per_row: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;
@group(0) @binding(3) var<uniform> params: Params;

fn byte_at(offset: u32) -> u32 {
  let word = weights[offset >> 2u];
  return (word >> ((offset & 3u) * 8u)) & 255u;
}

fn half_at(offset: u32) -> f32 {
  let bits = byte_at(offset) | (byte_at(offset + 1u) << 8u);
  return unpack2x16float(bits).x;
}

`;

const q4Dequant = /* wgsl */`
fn dequant(row: u32, column: u32) -> f16 {
  let block_index = row * params.blocks_per_row + column / 256u;
  let block = block_index * 144u;
  let within = column & 255u;
  let group = within / 64u;
  let in_group = within & 63u;
  let scale_index = group * 2u + select(0u, 1u, in_group >= 32u);
  let scale_base = block + 4u;
  var scale: u32;
  var minimum: u32;
  if (scale_index < 4u) {
    scale = byte_at(scale_base + scale_index) & 63u;
    minimum = byte_at(scale_base + scale_index + 4u) & 63u;
  } else {
    scale = (byte_at(scale_base + scale_index + 4u) & 15u) |
      ((byte_at(scale_base + scale_index - 4u) >> 6u) << 4u);
    minimum = (byte_at(scale_base + scale_index + 4u) >> 4u) |
      ((byte_at(scale_base + scale_index) >> 6u) << 4u);
  }
  let packed = byte_at(block + 16u + group * 32u + (in_group & 31u));
  let quant = select(packed & 15u, packed >> 4u, in_group >= 32u);
  return f16(half_at(block) * f32(scale) * f32(quant) - half_at(block + 2u) * f32(minimum));
}

fn dequant4(row: u32, column: u32) -> vec4<f16> {
  let block_index = row * params.blocks_per_row + column / 256u;
  let block = block_index * 144u;
  let within = column & 255u;
  let group = within / 64u;
  let in_group = within & 63u;
  let scale_index = group * 2u + select(0u, 1u, in_group >= 32u);
  let scale_base = block + 4u;
  var scale: u32;
  var minimum: u32;
  if (scale_index < 4u) {
    scale = byte_at(scale_base + scale_index) & 63u;
    minimum = byte_at(scale_base + scale_index + 4u) & 63u;
  } else {
    scale = (byte_at(scale_base + scale_index + 4u) & 15u) |
      ((byte_at(scale_base + scale_index - 4u) >> 6u) << 4u);
    minimum = (byte_at(scale_base + scale_index + 4u) >> 4u) |
      ((byte_at(scale_base + scale_index) >> 6u) << 4u);
  }
  let quant_word = weights[(block + 16u + group * 32u + (in_group & 31u)) >> 2u];
  let packed = unpack4xU8(quant_word);
  let quants = select(packed & vec4<u32>(15u), packed >> vec4<u32>(4u), vec4<bool>(in_group >= 32u));
  let values = half_at(block) * f32(scale) * vec4<f32>(quants) - half_at(block + 2u) * f32(minimum);
  return vec4<f16>(values);
}
`;

const q40Dequant = /* wgsl */`
fn dequant4(row:u32,column:u32)->vec4<f16>{
  let block=(row*params.blocks_per_row+column/32u)*18u;let within=column&31u;
  let qoffset=block+2u+(within&15u);
  let packed=vec4<u32>(byte_at(qoffset),byte_at(qoffset+1u),byte_at(qoffset+2u),byte_at(qoffset+3u));
  let quants=select(packed&vec4<u32>(15u),packed>>vec4<u32>(4u),vec4<bool>(within>=16u));
  return vec4<f16>(half_at(block)*(vec4<f32>(quants)-vec4<f32>(8.0)));
}
`;

const q6Dequant = /* wgsl */`
fn dequant(row: u32, column: u32) -> f16 {
  let block_index = row * params.blocks_per_row + column / 256u;
  let block = block_index * 210u;
  let within = column & 255u;
  let half = within / 128u;
  let position = within & 127u;
  let quadrant = position / 32u;
  let lane = position & 31u;
  let low_byte = byte_at(block + half * 64u + lane + (quadrant & 1u) * 32u);
  let low = select(low_byte & 15u, low_byte >> 4u, quadrant >= 2u);
  let high_byte = byte_at(block + 128u + half * 32u + lane);
  let high = (high_byte >> (quadrant * 2u)) & 3u;
  let quant = i32(low | (high << 4u)) - 32;
  let scale_byte = byte_at(block + 192u + half * 8u + (lane / 16u) + quadrant * 2u);
  let scale = select(i32(scale_byte), i32(scale_byte) - 256, scale_byte >= 128u);
  return f16(half_at(block + 208u) * f32(scale * quant));
}

fn dequant4(row: u32, column: u32) -> vec4<f16> {
  let block_index = row * params.blocks_per_row + column / 256u;
  let block = block_index * 210u;
  let within = column & 255u;
  let half = within / 128u;
  let position = within & 127u;
  let quadrant = position / 32u;
  let lane = position & 31u;
  let low_bytes = vec4<u32>(
    byte_at(block + half * 64u + lane + (quadrant & 1u) * 32u),
    byte_at(block + half * 64u + lane + (quadrant & 1u) * 32u + 1u),
    byte_at(block + half * 64u + lane + (quadrant & 1u) * 32u + 2u),
    byte_at(block + half * 64u + lane + (quadrant & 1u) * 32u + 3u)
  );
  let lows = select(low_bytes & vec4<u32>(15u), low_bytes >> vec4<u32>(4u), vec4<bool>(quadrant >= 2u));
  let high_bytes = vec4<u32>(
    byte_at(block + 128u + half * 32u + lane), byte_at(block + 128u + half * 32u + lane + 1u),
    byte_at(block + 128u + half * 32u + lane + 2u), byte_at(block + 128u + half * 32u + lane + 3u)
  );
  let highs = (high_bytes >> vec4<u32>(quadrant * 2u)) & vec4<u32>(3u);
  let quants = vec4<i32>(lows | (highs << vec4<u32>(4u))) - vec4<i32>(32);
  let scale_byte = byte_at(block + 192u + half * 8u + (lane / 16u) + quadrant * 2u);
  let scale = select(i32(scale_byte), i32(scale_byte) - 256, scale_byte >= 128u);
  return vec4<f16>(half_at(block + 208u) * f32(scale) * vec4<f32>(quants));
}
`;

function createShaderMain(tileK: 64 | 128, halfAccumulation = false): string {
  const vectorK = tileK / 4;
  const tileElements = 16 * vectorK;
  const accumulatorType = halfAccumulation ? 'f16' : 'f32';
  const dotExpression = (left: string, right: string) => halfAccumulation ? `dot(${left}, ${right})` : `f32(dot(${left}, ${right}))`;
  return /* wgsl */`
var<workgroup> tile_a: array<vec4<f16>, ${tileElements}>;
var<workgroup> tile_w: array<vec4<f16>, ${tileElements}>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_index: u32,
) {
  let global_m0 = workgroup_id.y * 16u + local_id.y;
  let global_m1 = global_m0 + 8u;
  let global_n0 = workgroup_id.x * 16u + local_id.x;
  let global_n1 = global_n0 + 8u;
  var sums = vec4<${accumulatorType}>(0.0);
  var base_k = 0u;
  loop {
    if (base_k >= params.k) { break; }
    var index = local_index;
    loop {
      if (index >= ${tileElements}u) { break; }
      let tile_row = index / ${vectorK}u;
      let tile_k = (index % ${vectorK}u) * 4u;
      let input_row = workgroup_id.y * 16u + tile_row;
      let weight_row = workgroup_id.x * 16u + tile_row;
      tile_a[index] = select(vec4<f16>(0.0), input[(input_row * params.k + base_k + tile_k) / 4u], input_row < params.m);
      tile_w[index] = select(vec4<f16>(0.0), dequant4(weight_row, base_k + tile_k), weight_row < params.n);
      index += 64u;
    }
    workgroupBarrier();
    if (global_m0 < params.m && global_n0 < params.n) {
      let a0 = local_id.y * ${vectorK}u;
      let a1 = (local_id.y + 8u) * ${vectorK}u;
      let w0 = local_id.x * ${vectorK}u;
      let w1 = (local_id.x + 8u) * ${vectorK}u;
      for (var k = 0u; k < ${vectorK}u; k += 1u) {
        let av0 = tile_a[a0 + k];
        let av1 = tile_a[a1 + k];
        let wv0 = tile_w[w0 + k];
        let wv1 = tile_w[w1 + k];
        sums += vec4<${accumulatorType}>(
          ${dotExpression('av0', 'wv0')}, ${dotExpression('av0', 'wv1')},
          ${dotExpression('av1', 'wv0')}, ${dotExpression('av1', 'wv1')}
        );
      }
    }
    workgroupBarrier();
    base_k += ${tileK}u;
  }
  if (global_m0 < params.m && global_n0 < params.n) { output[global_m0 * params.n + global_n0] = f16(sums.x); }
  if (global_m0 < params.m && global_n1 < params.n) { output[global_m0 * params.n + global_n1] = f16(sums.y); }
  if (global_m1 < params.m && global_n0 < params.n) { output[global_m1 * params.n + global_n0] = f16(sums.z); }
  if (global_m1 < params.m && global_n1 < params.n) { output[global_m1 * params.n + global_n1] = f16(sums.w); }
}
`;
}

function createBatchShaderMain(): string {
  return /* wgsl */`
var<workgroup> tile_a: array<vec4<f16>, 512>;
var<workgroup> tile_w: array<vec4<f16>, 512>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let m0=group.y*32u+local.y; let n0=group.x*32u+local.x;
  var sums0=vec4<f16>(0.0); var sums1=vec4<f16>(0.0); var sums2=vec4<f16>(0.0); var sums3=vec4<f16>(0.0);
  for (var base_k=0u; base_k<params.k; base_k+=64u) {
    var index=lane;
    loop {
      if (index>=1024u) { break; }
      let is_weight=index>=512u; let tile_index=index%512u; let row=tile_index/16u; let k4=tile_index%16u;
      if (!is_weight) {
        let global_m=group.y*32u+row;
        tile_a[tile_index]=select(vec4<f16>(0.0),input[(global_m*params.k+base_k+k4*4u)/4u],global_m<params.m);
      } else {
        let global_n=group.x*32u+row;
        tile_w[tile_index]=select(vec4<f16>(0.0),dequant4(global_n,base_k+k4*4u),global_n<params.n);
      }
      index+=64u;
    }
    workgroupBarrier();
    let a0=local.y*16u; let a1=(local.y+8u)*16u; let a2=(local.y+16u)*16u; let a3=(local.y+24u)*16u;
    let w0=local.x*16u; let w1=(local.x+8u)*16u; let w2=(local.x+16u)*16u; let w3=(local.x+24u)*16u;
    for (var k4=0u;k4<16u;k4+=1u) {
      let b0=tile_w[w0+k4]; let b1=tile_w[w1+k4]; let b2=tile_w[w2+k4]; let b3=tile_w[w3+k4];
      let av0=tile_a[a0+k4]; let av1=tile_a[a1+k4]; let av2=tile_a[a2+k4]; let av3=tile_a[a3+k4];
      sums0+=vec4<f16>(dot(av0,b0),dot(av0,b1),dot(av0,b2),dot(av0,b3));
      sums1+=vec4<f16>(dot(av1,b0),dot(av1,b1),dot(av1,b2),dot(av1,b3));
      sums2+=vec4<f16>(dot(av2,b0),dot(av2,b1),dot(av2,b2),dot(av2,b3));
      sums3+=vec4<f16>(dot(av3,b0),dot(av3,b1),dot(av3,b2),dot(av3,b3));
    }
    workgroupBarrier();
  }
  let rows=array<u32,4>(m0,m0+8u,m0+16u,m0+24u); let cols=array<u32,4>(n0,n0+8u,n0+16u,n0+24u);
  let all_sums=array<vec4<f16>,4>(sums0,sums1,sums2,sums3);
  for(var r=0u;r<4u;r+=1u) { for(var c=0u;c<4u;c+=1u) { if(rows[r]<params.m&&cols[c]<params.n) { output[rows[r]*params.n+cols[c]]=f16(all_sums[r][c]); } } }
}
`;
}

function createMidBatchShaderMain(): string {
  return /* wgsl */`
var<workgroup> tile_a: array<vec4<f16>, 256>;
var<workgroup> tile_w: array<vec4<f16>, 512>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let m0=group.y*16u+local.y; let m1=m0+8u;
  let n0=group.x*32u+local.x;
  var sums0=vec4<f16>(0.0); var sums1=vec4<f16>(0.0);
  for (var base_k=0u; base_k<params.k; base_k+=64u) {
    var index=lane;
    loop {
      if (index>=768u) { break; }
      let is_weight=index>=256u;
      let tile_index=select(index,index-256u,is_weight);
      let row=tile_index/16u; let k4=tile_index%16u;
      if (!is_weight) {
        let global_m=group.y*16u+row;
        tile_a[tile_index]=select(vec4<f16>(0.0),input[(global_m*params.k+base_k+k4*4u)/4u],global_m<params.m);
      } else {
        let global_n=group.x*32u+row;
        tile_w[tile_index]=select(vec4<f16>(0.0),dequant4(global_n,base_k+k4*4u),global_n<params.n);
      }
      index+=64u;
    }
    workgroupBarrier();
    let a0=local.y*16u; let a1=(local.y+8u)*16u;
    let w0=local.x*16u; let w1=(local.x+8u)*16u; let w2=(local.x+16u)*16u; let w3=(local.x+24u)*16u;
    for (var k4=0u;k4<16u;k4+=1u) {
      let av0=tile_a[a0+k4]; let av1=tile_a[a1+k4];
      let b0=tile_w[w0+k4]; let b1=tile_w[w1+k4]; let b2=tile_w[w2+k4]; let b3=tile_w[w3+k4];
      sums0+=vec4<f16>(dot(av0,b0),dot(av0,b1),dot(av0,b2),dot(av0,b3));
      sums1+=vec4<f16>(dot(av1,b0),dot(av1,b1),dot(av1,b2),dot(av1,b3));
    }
    workgroupBarrier();
  }
  let cols=array<u32,4>(n0,n0+8u,n0+16u,n0+24u);
  for(var c=0u;c<4u;c+=1u) {
    if(m0<params.m&&cols[c]<params.n) { output[m0*params.n+cols[c]]=sums0[c]; }
    if(m1<params.m&&cols[c]<params.n) { output[m1*params.n+cols[c]]=sums1[c]; }
  }
}
`;
}

export interface QuantMatmulRun {
  input: GPUBuffer;
  weights: GPUBuffer;
  output: GPUBuffer;
  bindGroup: GPUBindGroup;
  pipeline: GPUComputePipeline;
  m: number;
  n: number;
  k: number;
}

export class QuantMatmulKernel {
  readonly latencyPipeline: GPUComputePipeline;
  readonly throughputPipeline: GPUComputePipeline;
  readonly midBatchPipeline: GPUComputePipeline;
  readonly batchPipeline: GPUComputePipeline;

  constructor(readonly device: GPUDevice, readonly type: GGMLType.Q4_0 | GGMLType.Q4_K | GGMLType.Q6_K) {
    this.latencyPipeline = this.createPipeline(128, 'latency');
    this.throughputPipeline = this.createPipeline(64, 'throughput');
    this.midBatchPipeline = this.createMidBatchPipeline();
    this.batchPipeline = this.createBatchPipeline();
  }

  private createMidBatchPipeline(): GPUComputePipeline {
    const dequant=this.type===GGMLType.Q4_0?q40Dequant:this.type===GGMLType.Q4_K?q4Dequant:q6Dequant;
    const code=shaderPrelude+dequant+createMidBatchShaderMain();
    const quantName=this.type===GGMLType.Q4_0?'Q4_0':this.type===GGMLType.Q4_K?'Q4_K':'Q6_K';
    const module=this.device.createShaderModule({code,label:`${quantName} mid-batch shader`});
    return this.device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'main'},label:`${quantName} fused matmul (mid-batch)`});
  }

  private createBatchPipeline(): GPUComputePipeline {
    const dequant=this.type===GGMLType.Q4_0?q40Dequant:this.type===GGMLType.Q4_K?q4Dequant:q6Dequant;
    const code=shaderPrelude+dequant+createBatchShaderMain();
    const quantName=this.type===GGMLType.Q4_0?'Q4_0':this.type===GGMLType.Q4_K?'Q4_K':'Q6_K';
    const module=this.device.createShaderModule({code,label:`${quantName} batch shader`});
    return this.device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'main'},label:`${quantName} fused matmul (batch)`});
  }

  private createPipeline(tileK: 64 | 128, variant: string, halfAccumulation = false): GPUComputePipeline {
    const dequant=this.type===GGMLType.Q4_0?q40Dequant:this.type===GGMLType.Q4_K?q4Dequant:q6Dequant;
    const code = shaderPrelude + dequant + createShaderMain(tileK, halfAccumulation);
    const quantName = this.type===GGMLType.Q4_0?'Q4_0':this.type === GGMLType.Q4_K ? 'Q4_K' : 'Q6_K';
    const module = this.device.createShaderModule({ code, label: `${quantName} ${variant} shader` });
    return this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
      label: `${quantName} fused matmul (${variant})`,
    });
  }

  createRun(input: Uint16Array, weights: Uint8Array, m: number, n: number, k: number): QuantMatmulRun {
    if (k % TILE_K !== 0) throw new Error('quantized matmul K must be divisible by 256');
    if (input.length !== m * k) throw new Error(`input has ${input.length} values; expected ${m * k}`);
    const inputBuffer = createBufferWithData(this.device, input, GPUBufferUsage.STORAGE, 'matmul input');
    const weightBuffer = createBufferWithData(this.device, weights, GPUBufferUsage.STORAGE, 'quantized weights');
    return this.createRunFromBuffers(inputBuffer, weightBuffer, m, n, k);
  }

  createRunFromBuffers(inputBuffer: GPUBuffer, weightBuffer: GPUBuffer, m: number, n: number, k: number, outputBuffer?: GPUBuffer): QuantMatmulRun {
    if (k % TILE_K !== 0) throw new Error('quantized matmul K must be divisible by 256');
    const output = outputBuffer ?? this.device.createBuffer({
      size: Math.ceil(m * n * 2 / 4) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'matmul output',
    });
    const params = createBufferWithData(
      this.device,
      new Uint32Array([m, n, k, k / (this.type === GGMLType.Q4_0 ? 32 : 256)]),
      GPUBufferUsage.UNIFORM,
      'matmul params',
    );
    const pipeline = m >= 512 ? this.batchPipeline : m >= 256 ? this.midBatchPipeline : m >= 64 ? this.throughputPipeline : this.latencyPipeline;
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    return { input: inputBuffer, weights: weightBuffer, output, bindGroup, pipeline, m, n, k };
  }

  encode(pass: GPUComputePassEncoder, run: QuantMatmulRun): void {
    pass.setPipeline(run.pipeline);
    pass.setBindGroup(0, run.bindGroup);
    const batch = run.pipeline === this.batchPipeline;
    const midBatch = run.pipeline === this.midBatchPipeline;
    pass.dispatchWorkgroups(Math.ceil(run.n / (batch || midBatch ? 32 : TILE_N)), Math.ceil(run.m / (batch ? 32 : TILE_M)));
  }

  dispatch(run: QuantMatmulRun, repetitions = 1): void {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (let index = 0; index < repetitions; index += 1) this.encode(pass, run);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
