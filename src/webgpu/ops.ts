import { createBufferWithData } from './device.ts';

function paramsBuffer(device: GPUDevice, bytes: number, write: (view: DataView) => void): GPUBuffer {
  const data = new ArrayBuffer(bytes);
  write(new DataView(data));
  return createBufferWithData(device, new Uint8Array(data), GPUBufferUsage.UNIFORM, 'kernel params');
}

export interface EncodableRun {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  dispatch: [number, number?, number?];
}

export function encodeRun(pass: GPUComputePassEncoder, run: EncodableRun): void {
  pass.setPipeline(run.pipeline);
  pass.setBindGroup(0, run.bindGroup);
  pass.dispatchWorkgroups(run.dispatch[0], run.dispatch[1] ?? 1, run.dispatch[2] ?? 1);
}

const rmsNormShader = /* wgsl */`
enable f16;
enable subgroups;
struct Params { rows: u32, width: u32, residual: u32, _pad: u32, epsilon: f32 }
@group(0) @binding(0) var<storage, read_write> x: array<f16>;
@group(0) @binding(1) var<storage, read> residual: array<f16>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> normalized: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> sums: array<f32, 128>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>, @builtin(subgroup_invocation_id) subgroup_lane: u32, @builtin(subgroup_id) subgroup_id: u32, @builtin(num_subgroups) num_subgroups: u32) {
  let row = group.x;
  let lane = local.x;
  var sum = 0.0;
  for (var column = lane; column < params.width; column += 256u) {
    let index = row * params.width + column;
    var value = f32(x[index]);
    if (params.residual != 0u) { value += f32(residual[index]); x[index] = f16(value); }
    sum += value * value;
  }
  let partial=subgroupAdd(sum);
  if(subgroup_lane==0u){sums[subgroup_id]=partial;}
  workgroupBarrier();
  if(subgroup_id==0u){
    let value=select(0.0,sums[subgroup_lane],subgroup_lane<num_subgroups);
    let total=subgroupAdd(value);
    if(subgroup_lane==0u){sums[0]=total;}
  }
  workgroupBarrier();
  let inv_rms = inverseSqrt(sums[0] / f32(params.width) + params.epsilon);
  for (var column = lane; column < params.width; column += 256u) {
    let index = row * params.width + column;
    normalized[index] = f16(f32(x[index]) * inv_rms * weight[column]);
  }
}
`;

export class RmsNormKernel {
  readonly pipeline: GPUComputePipeline;
  constructor(readonly device: GPUDevice) {
    this.pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: rmsNormShader }), entryPoint: 'main' }, label: 'residual + RMSNorm' });
  }
  createRun(x: GPUBuffer, residual: GPUBuffer, weight: GPUBuffer, normalized: GPUBuffer, rows: number, width: number, epsilon: number, hasResidual: boolean): EncodableRun {
    const params = paramsBuffer(this.device, 32, (v) => { v.setUint32(0, rows, true); v.setUint32(4, width, true); v.setUint32(8, hasResidual ? 1 : 0, true); v.setFloat32(16, epsilon, true); });
    return { pipeline: this.pipeline, bindGroup: this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: x } }, { binding: 1, resource: { buffer: residual } },
      { binding: 2, resource: { buffer: weight } }, { binding: 3, resource: { buffer: normalized } },
      { binding: 4, resource: { buffer: params } },
    ] }), dispatch: [rows] };
  }
}

const swigluShader = /* wgsl */`
enable f16;
struct Params { rows: u32, width: u32 }
@group(0) @binding(0) var<storage, read> combined: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row=id.y; let column=id.x; let vectors_per_row=params.width/4u;
  if(row>=params.rows||column>=vectors_per_row){return;}
  let index=row*vectors_per_row+column;
  let g = vec4<f32>(combined[row*vectors_per_row*2u+column]);
  let silu = g / (vec4<f32>(1.0) + exp(-g));
  output[index] = vec4<f16>(silu * vec4<f32>(combined[row*vectors_per_row*2u+vectors_per_row+column]));
}
`;

export class SwiGLUKernel {
  readonly pipeline: GPUComputePipeline;
  constructor(readonly device: GPUDevice) {
    this.pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: swigluShader }), entryPoint: 'main' }, label: 'fused SwiGLU' });
  }
  createRun(combined: GPUBuffer, output: GPUBuffer, rows: number, width: number): EncodableRun {
    const params = createBufferWithData(this.device, new Uint32Array([rows, width, 0, 0]), GPUBufferUsage.UNIFORM);
    return { pipeline: this.pipeline, bindGroup: this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: combined } }, { binding: 1, resource: { buffer: output } }, { binding: 2, resource: { buffer: params } },
    ] }), dispatch: [Math.ceil(width / 4 / 256), rows] };
  }
}

const qkNormRopeShader = /* wgsl */`
enable f16;
enable subgroups;
struct Params { tokens: u32, sequence: u32, input_stride: u32, q_heads: u32, kv_heads: u32, dispatch_width: u32, epsilon: f32, theta: f32 }
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> q_weight: array<f32>;
@group(0) @binding(2) var<storage, read> k_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> q_output: array<f16>;
@group(0) @binding(4) var<storage, read_write> k_output: array<f16>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read> rope_table: array<vec2<f32>>;
var<workgroup> values: array<f32, 128>;
var<workgroup> sums: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>, @builtin(subgroup_invocation_id) subgroup_lane: u32, @builtin(subgroup_id) subgroup_id: u32, @builtin(num_subgroups) num_subgroups: u32) {
  let linear_group=group.x+group.y*params.dispatch_width;
  let total_heads=params.q_heads+params.kv_heads;
  if(linear_group>=params.tokens*total_heads){return;}
  let token=linear_group/total_heads; let combined_head=linear_group%total_heads;
  let is_q=combined_head<params.q_heads; let head=select(combined_head-params.q_heads,combined_head,is_q);
  let heads=select(params.kv_heads,params.q_heads,is_q);
  let d = local.x;
  let input_offset=select(params.q_heads*128u,0u,is_q);
  let input_base = token * params.input_stride + input_offset + head * 128u;
  let output_base = (token * heads + head) * 128u;
  values[d] = f32(input[input_base + d]);
  let partial=subgroupAdd(values[d]*values[d]);
  if(subgroup_lane==0u){sums[subgroup_id]=partial;}
  workgroupBarrier();
  if(subgroup_id==0u){
    let value=select(0.0,sums[subgroup_lane],subgroup_lane<num_subgroups);
    let total=subgroupAdd(value);
    if(subgroup_lane==0u){sums[0]=total;}
  }
  workgroupBarrier();
  let norm_weight=select(k_weight[d],q_weight[d],is_q);
  let normed = values[d] * inverseSqrt(sums[0] / 128.0 + params.epsilon) * norm_weight;
  values[d] = normed;
  workgroupBarrier();
  let rotary_d = d % 64u;
  let partner = select(d + 64u, d - 64u, d >= 64u);
  let rotated = select(-values[partner], values[partner], d >= 64u);
  let rope=rope_table[(token%params.sequence)*64u+rotary_d];
  let result=f16(normed*rope.x+rotated*rope.y);
  if(is_q){q_output[output_base+d]=result;}else{k_output[output_base+d]=result;}
}
`;

export class QKNormRopeKernel {
  readonly pipeline: GPUComputePipeline;
  constructor(readonly device: GPUDevice) {
    this.pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: qkNormRopeShader }), entryPoint: 'main' }, label: 'QK norm + RoPE' });
  }
  createRun(input: GPUBuffer, qWeight: GPUBuffer, kWeight: GPUBuffer, qOutput: GPUBuffer, kOutput: GPUBuffer, tokens: number, sequence: number, epsilon: number, ropeTable: GPUBuffer, inputStride = 3072): EncodableRun {
    const groups = tokens * 24;
    const dispatchWidth = Math.min(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
    const params = paramsBuffer(this.device, 32, (v) => { v.setUint32(0,tokens,true);v.setUint32(4,sequence,true);v.setUint32(8,inputStride,true);v.setUint32(12,16,true);v.setUint32(16,8,true);v.setUint32(20,dispatchWidth,true);v.setFloat32(24,epsilon,true);v.setFloat32(28,0,true); });
    return { pipeline: this.pipeline, bindGroup: this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      {binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:qWeight}},{binding:2,resource:{buffer:kWeight}},
      {binding:3,resource:{buffer:qOutput}},{binding:4,resource:{buffer:kOutput}},{binding:5,resource:{buffer:params}},{binding:6,resource:{buffer:ropeTable}},
    ] }), dispatch: [dispatchWidth, Math.ceil(groups / dispatchWidth)] };
  }
}

const tiledFlashAttentionShader = /* wgsl */`
enable f16;
struct Params { batch: u32, sequence: u32, q_heads: u32, kv_heads: u32, value_stride: u32, value_offset: u32 }
@group(0) @binding(0) var<storage, read> query: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> key: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> value: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> query_tile:array<vec4<f16>,512>;
var<workgroup> key_tile:array<vec4<f16>,512>;
var<workgroup> value_tile:array<vec4<f16>,512>;
var<workgroup> total_tile:array<vec4<f16>,512>;
var<workgroup> scores:array<f32,256>;
var<workgroup> weights:array<f16,256>;
var<workgroup> maxima:array<f32,16>;
var<workgroup> denominators:array<f16,16>;
var<workgroup> alphas:array<f16,16>;
var<workgroup> betas:array<f16,16>;
fn update_stats(row:u32){
  var block_max=-3.402823e38;
  for(var column=0u;column<16u;column+=1u){block_max=max(block_max,scores[row*16u+column]);}
  var block_den=0.0;
  for(var column=0u;column<16u;column+=1u){let weight=exp(scores[row*16u+column]-block_max);weights[row*16u+column]=f16(weight);block_den+=weight;}
  let next_max=max(maxima[row],block_max);let alpha=exp(maxima[row]-next_max);let beta=exp(block_max-next_max);
  alphas[row]=f16(alpha);betas[row]=f16(beta);denominators[row]=denominators[row]*f16(alpha)+f16(beta*block_den);maxima[row]=next_max;
}
@compute @workgroup_size(16,16,1)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let batch=group.y/params.q_heads;let q_head=group.y%params.q_heads;let kv_head=q_head/(params.q_heads/params.kv_heads);
  let q_pos=group.x*16u+local.y;
  for(var index=lane;index<512u;index+=256u){
    let row=index/32u;let d4=index%32u;let position=group.x*16u+row;
    query_tile[index]=select(vec4<f16>(0.0),query[((batch*params.sequence+position)*params.q_heads+q_head)*32u+d4],position<params.sequence);
    total_tile[index]=vec4<f16>(0.0);
  }
  if(local.x==0u){maxima[local.y]=-3.402823e38;denominators[local.y]=f16(0.0);}
  workgroupBarrier();
  let key_limit=min(params.sequence,(group.x+1u)*16u);
  let prefix_size=min(params.sequence,176u);let local_window=min(params.sequence,432u);let query_tile_start=group.x*16u;
  let local_start=select(0u,query_tile_start+1u-local_window,query_tile_start+1u>local_window);
  for(var key_base=0u;key_base<key_limit;key_base+=16u){
    if(key_base>=prefix_size&&key_base+15u<local_start){continue;}
    for(var index=lane;index<512u;index+=256u){
      let row=index/32u;let d4=index%32u;let k_pos=key_base+row;
      if(k_pos<params.sequence){
        key_tile[index]=key[((batch*params.sequence+k_pos)*params.kv_heads+kv_head)*32u+d4];
        let scalar_offset=(batch*params.sequence+k_pos)*params.value_stride+params.value_offset+kv_head*128u;
        value_tile[index]=value[scalar_offset/4u+d4];
      }else{key_tile[index]=vec4<f16>(0.0);value_tile[index]=vec4<f16>(0.0);}
    }
    workgroupBarrier();
    let k_pos=key_base+local.x;var score=-3.402823e38;
    if(q_pos<params.sequence&&k_pos<=q_pos&&k_pos<params.sequence&&(k_pos<prefix_size||k_pos+local_window>q_pos)){
      score=0.0;
      for(var d4=0u;d4<32u;d4+=1u){score+=f32(dot(query_tile[local.y*32u+d4],key_tile[local.x*32u+d4]));}
      score*=0.08838834764831845;
    }
    scores[local.y*16u+local.x]=score;
    workgroupBarrier();
    if(local.x==0u){update_stats(local.y);}
    workgroupBarrier();
    for(var index=lane;index<512u;index+=256u){
      let row=index/32u;let d4=index%32u;var av=vec4<f16>(0.0);
      for(var column=0u;column<16u;column+=1u){av+=weights[row*16u+column]*value_tile[column*32u+d4];}
      total_tile[index]=total_tile[index]*alphas[row]+betas[row]*av;
    }
    workgroupBarrier();
  }
  for(var index=lane;index<512u;index+=256u){
    let row=index/32u;let d4=index%32u;let position=group.x*16u+row;
    if(position<params.sequence){output[((batch*params.sequence+position)*params.q_heads+q_head)*32u+d4]=vec4<f16>(total_tile[index]/denominators[row]);}
  }
}
`;

const chunkedQKShader = /* wgsl */`
enable f16;
struct Params { batch: u32, sequence: u32, q_heads: u32, kv_heads: u32, value_stride: u32, value_offset: u32, q_offset: u32, q_count: u32 }
@group(0) @binding(0) var<storage, read> query: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> key: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> scores: array<f16>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> q_tile: array<vec4<f16>, 512>;
var<workgroup> k_tile: array<vec4<f16>, 512>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let local_q_base=group.y*16u;
  if(local_q_base>=params.q_count || group.x*16u>params.q_offset+local_q_base+15u){return;}
  let batch=group.z/params.q_heads;let q_head=group.z%params.q_heads;
  let kv_head=q_head/(params.q_heads/params.kv_heads);
  var index=lane;
  loop {
    if(index>=1024u){break;}
    let is_key=index>=512u;let tile_index=index%512u;
    let row=tile_index/32u;let d4=tile_index%32u;
    if(!is_key){
      let local_q=local_q_base+row;let pos=params.q_offset+local_q;
      q_tile[tile_index]=select(vec4<f16>(0.0),query[((batch*params.sequence+pos)*params.q_heads+q_head)*32u+d4],local_q<params.q_count);
    }else{
      let pos=group.x*16u+row;
      k_tile[tile_index]=select(vec4<f16>(0.0),key[((batch*params.sequence+pos)*params.kv_heads+kv_head)*32u+d4],pos<params.sequence);
    }
    index+=64u;
  }
  workgroupBarrier();
  let local_q0=local_q_base+local.y;let local_q1=local_q0+8u;
  let q0=params.q_offset+local_q0;let q1=params.q_offset+local_q1;
  let k0=group.x*16u+local.x;let k1=k0+8u;
  let qa0=local.y*32u;let qa1=(local.y+8u)*32u;
  let kb0=local.x*32u;let kb1=(local.x+8u)*32u;
  var sums=vec4<f32>(0.0);
  for(var d4=0u;d4<32u;d4+=1u){
    let a0=q_tile[qa0+d4];let a1=q_tile[qa1+d4];let b0=k_tile[kb0+d4];let b1=k_tile[kb1+d4];
    sums+=vec4<f32>(f32(dot(a0,b0)),f32(dot(a0,b1)),f32(dot(a1,b0)),f32(dot(a1,b1)));
  }
  let score_base=(batch*params.q_heads+q_head)*64u*params.sequence;
  if(local_q0<params.q_count&&k0<=q0){scores[score_base+local_q0*params.sequence+k0]=f16(sums.x*0.08838834764831845);}
  if(local_q0<params.q_count&&k1<=q0){scores[score_base+local_q0*params.sequence+k1]=f16(sums.y*0.08838834764831845);}
  if(local_q1<params.q_count&&k0<=q1){scores[score_base+local_q1*params.sequence+k0]=f16(sums.z*0.08838834764831845);}
  if(local_q1<params.q_count&&k1<=q1){scores[score_base+local_q1*params.sequence+k1]=f16(sums.w*0.08838834764831845);}
}
`;

const chunkedAttentionValueShader = /* wgsl */`
enable f16;
enable subgroups;
struct Params { batch: u32, sequence: u32, q_heads: u32, kv_heads: u32, value_stride: u32, value_offset: u32, q_offset: u32, q_count: u32 }
@group(0) @binding(0) var<storage, read> scores: array<f16>;
@group(0) @binding(1) var<storage, read> value: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) group: vec3<u32>,@builtin(local_invocation_id) local: vec3<u32>){
  let local_q=group.x;let q_pos=params.q_offset+local_q;
  let batch=group.y/params.q_heads;let q_head=group.y%params.q_heads;
  let kv_head=q_head/(params.q_heads/params.kv_heads);let d4=local.x;
  let score_base=(batch*params.q_heads+q_head)*64u*params.sequence+local_q*params.sequence;
  var total=vec4<f32>(0.0);var denominator=0.0;var maximum=-3.402823e38;
  for(var k_pos=0u;k_pos<=q_pos;k_pos+=1u){
    var own_score=f16(0.0);if(d4==0u){own_score=scores[score_base+k_pos];}
    let score=f32(subgroupBroadcastFirst(own_score));let next_max=max(maximum,score);
    var own_alpha=1.0;var own_beta=0.0;if(d4==0u){own_alpha=exp(maximum-next_max);own_beta=exp(score-next_max);}
    let alpha=subgroupBroadcastFirst(own_alpha);let beta=subgroupBroadcastFirst(own_beta);
    let scalar_offset=(batch*params.sequence+k_pos)*params.value_stride+params.value_offset+kv_head*128u;
    let v4=value[scalar_offset/4u+d4];total=total*alpha+beta*vec4<f32>(v4);denominator=denominator*alpha+beta;maximum=next_max;
  }
  output[((batch*params.sequence+q_pos)*params.q_heads+q_head)*32u+d4]=vec4<f16>(total/denominator);
}
`;

export interface AttentionRun { commands: EncodableRun[]; ownedBuffers: GPUBuffer[] }

export class CausalAttentionKernel {
  readonly tiledPipeline:GPUComputePipeline;readonly qkPipeline:GPUComputePipeline;readonly chunkedAvPipeline:GPUComputePipeline;
  constructor(readonly device: GPUDevice) {
    this.tiledPipeline=device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:tiledFlashAttentionShader}),entryPoint:'main'},label:'tiled flash attention'});
    this.qkPipeline=device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:chunkedQKShader}),entryPoint:'main'},label:'chunked tiled QK scores'});
    this.chunkedAvPipeline=device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:chunkedAttentionValueShader}),entryPoint:'main'},label:'chunked online-softmax attention values'});
  }
  createRun(q:GPUBuffer,k:GPUBuffer,v:GPUBuffer,output:GPUBuffer,batch:number,sequence:number,qHeads=16,kvHeads=8,valueStride=1024,valueOffset=0):AttentionRun {
    if(sequence<=8192){
      const common=createBufferWithData(this.device,new Uint32Array([batch,sequence,qHeads,kvHeads,valueStride,valueOffset,0,0]),GPUBufferUsage.UNIFORM);
      const fused:EncodableRun={pipeline:this.tiledPipeline,bindGroup:this.device.createBindGroup({layout:this.tiledPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:q}},{binding:1,resource:{buffer:k}},{binding:2,resource:{buffer:v}},{binding:3,resource:{buffer:output}},{binding:4,resource:{buffer:common}}]}),dispatch:[Math.ceil(sequence/16),batch*qHeads]};
      return {commands:[fused],ownedBuffers:[common]};
    }
    const scores=this.device.createBuffer({size:batch*qHeads*64*sequence*2,usage:GPUBufferUsage.STORAGE,label:'chunked attention score workspace'});
    const commands:EncodableRun[]=[];const ownedBuffers:GPUBuffer[]=[scores];
    for(let qOffset=0;qOffset<sequence;qOffset+=64){
      const qCount=Math.min(64,sequence-qOffset);
      const common=createBufferWithData(this.device,new Uint32Array([batch,sequence,qHeads,kvHeads,valueStride,valueOffset,qOffset,qCount]),GPUBufferUsage.UNIFORM);ownedBuffers.push(common);
      const qk:EncodableRun={pipeline:this.qkPipeline,bindGroup:this.device.createBindGroup({layout:this.qkPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:q}},{binding:1,resource:{buffer:k}},{binding:2,resource:{buffer:scores}},{binding:3,resource:{buffer:common}}]}),dispatch:[Math.ceil(sequence/16),Math.ceil(qCount/16),batch*qHeads]};
      const av:EncodableRun={pipeline:this.chunkedAvPipeline,bindGroup:this.device.createBindGroup({layout:this.chunkedAvPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:scores}},{binding:1,resource:{buffer:v}},{binding:2,resource:{buffer:output}},{binding:3,resource:{buffer:common}}]}),dispatch:[qCount,batch*qHeads]};
      commands.push(qk,av);
    }
    return {commands,ownedBuffers};
  }
  encode(pass:GPUComputePassEncoder,run:AttentionRun):void { for(const command of run.commands)encodeRun(pass,command); }
}

const poolShader = /* wgsl */`
enable f16;
struct Params { batch: u32, sequence: u32, width: u32 }
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> lengths: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> sums: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let batch = group.x; let lane = local.x; let row = batch * params.sequence + lengths[batch] - 1u;
  var sum = 0.0;
  for (var d = lane; d < params.width; d += 256u) { let v = f32(input[row * params.width + d]); sum += v * v; }
  sums[lane] = sum; workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) { if (lane < stride) { sums[lane] += sums[lane + stride]; } workgroupBarrier(); }
  let inv = inverseSqrt(sums[0]);
  for (var d = lane; d < params.width; d += 256u) { output[batch * params.width + d] = f32(input[row * params.width + d]) * inv; }
}
`;

export class LastTokenPoolKernel {
  readonly pipeline: GPUComputePipeline;
  constructor(readonly device: GPUDevice) {
    this.pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: poolShader }), entryPoint: 'main' }, label: 'last-token L2 pool' });
  }
  createRun(input: GPUBuffer, lengths: GPUBuffer, output: GPUBuffer, batch: number, sequence: number, width = 1024): EncodableRun {
    const params = createBufferWithData(this.device, new Uint32Array([batch, sequence, width, 0]), GPUBufferUsage.UNIFORM);
    return { pipeline: this.pipeline, bindGroup: this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: lengths } },
      { binding: 2, resource: { buffer: output } }, { binding: 3, resource: { buffer: params } },
    ] }), dispatch: [batch] };
  }
}

const embeddingShader = /* wgsl */`
enable f16;
struct Params { tokens: u32, width: u32, blocks_per_row: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(3) var<uniform> params: Params;
fn byte_at(offset: u32) -> u32 { let word = weights[offset >> 2u]; return (word >> ((offset & 3u) * 8u)) & 255u; }
fn half_at(offset: u32) -> f32 { let bits = byte_at(offset) | (byte_at(offset + 1u) << 8u); return unpack2x16float(bits).x; }
fn dequant4(row: u32, column: u32) -> vec4<f16> {
  let block = (row * params.blocks_per_row + column / 256u) * 210u;
  let within = column & 255u; let half = within / 128u; let position = within & 127u;
  let quadrant = position / 32u; let lane = position & 31u;
  let low_offset = block + half * 64u + lane + (quadrant & 1u) * 32u;
  let low_bytes = vec4<u32>(byte_at(low_offset), byte_at(low_offset+1u), byte_at(low_offset+2u), byte_at(low_offset+3u));
  let lows = select(low_bytes & vec4<u32>(15u), low_bytes >> vec4<u32>(4u), vec4<bool>(quadrant >= 2u));
  let high_offset = block + 128u + half * 32u + lane;
  let high_bytes = vec4<u32>(byte_at(high_offset), byte_at(high_offset+1u), byte_at(high_offset+2u), byte_at(high_offset+3u));
  let highs = (high_bytes >> vec4<u32>(quadrant * 2u)) & vec4<u32>(3u);
  let quants = vec4<i32>(lows | (highs << vec4<u32>(4u))) - vec4<i32>(32);
  let scale_byte = byte_at(block + 192u + half * 8u + lane / 16u + quadrant * 2u);
  let scale = select(i32(scale_byte), i32(scale_byte) - 256, scale_byte >= 128u);
  return vec4<f16>(half_at(block + 208u) * f32(scale) * vec4<f32>(quants));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let vector_index = id.x; let vectors_per_token = params.width / 4u;
  if (vector_index >= params.tokens * vectors_per_token) { return; }
  let token_index = vector_index / vectors_per_token; let column = (vector_index % vectors_per_token) * 4u;
  output[vector_index] = dequant4(token_ids[token_index], column);
}
`;

export class EmbeddingLookupKernel {
  readonly pipeline: GPUComputePipeline;
  constructor(readonly device: GPUDevice) {
    this.pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: embeddingShader }), entryPoint: 'main' }, label: 'Q6_K embedding lookup' });
  }
  createRun(tokens: GPUBuffer, weights: GPUBuffer, output: GPUBuffer, tokenCount: number, width = 1024): EncodableRun {
    const params = createBufferWithData(this.device, new Uint32Array([tokenCount, width, width / 256, 0]), GPUBufferUsage.UNIFORM);
    return { pipeline: this.pipeline, bindGroup: this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: tokens } }, { binding: 1, resource: { buffer: weights } },
      { binding: 2, resource: { buffer: output } }, { binding: 3, resource: { buffer: params } },
    ] }), dispatch: [Math.ceil(tokenCount * width / 4 / 256)] };
  }
}
