import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { GGUFReader, QWEN3_METADATA_KEYS } from '../src/gguf/reader.ts';
import { GGMLType } from '../src/gguf/types.ts';
import { buildPrepackedFile, packQ40Tile32, type Q40TensorPart } from '../src/prepacked/format.ts';

const sourcePath = process.argv[2] ?? 'models/qwen3-embedding-0.6b-q4_0-webgpu.gguf';
const outputPath = process.argv[3] ?? 'models/qwen3-embedding-0.6b-q4_0-webgpu-tile32.wgpack';
const sourceFile = await readFile(sourcePath);
const sourceBuffer = sourceFile.buffer.slice(sourceFile.byteOffset, sourceFile.byteOffset + sourceFile.byteLength) as ArrayBuffer;
const gguf = new GGUFReader(sourceBuffer).parse({ metadataKeys: QWEN3_METADATA_KEYS });
const sourceSha256 = createHash('sha256').update(sourceFile).digest('hex');

function part(name: string): Q40TensorPart {
  const tensor = gguf.tensors.get(name);
  if (!tensor || tensor.type !== GGMLType.Q4_0 || tensor.dimensions.length !== 2) throw new Error(`${name} is not a Q4_0 matrix`);
  const [k, n] = tensor.dimensions;
  return { name, k, n, bytes: new Uint8Array(sourceBuffer, tensor.byteOffset, tensor.byteLength) };
}

const packed = [];
for (let layer = 0; layer < 28; layer += 1) {
  const prefix = `blk.${layer}`;
  packed.push(packQ40Tile32(`${prefix}.attn_qk.weight`, [part(`${prefix}.attn_q.weight`), part(`${prefix}.attn_k.weight`)]));
  packed.push(packQ40Tile32(`${prefix}.attn_v.weight`, [part(`${prefix}.attn_v.weight`)]));
  packed.push(packQ40Tile32(`${prefix}.attn_output.weight`, [part(`${prefix}.attn_output.weight`)]));
  packed.push(packQ40Tile32(`${prefix}.ffn_gate_up.weight`, [part(`${prefix}.ffn_gate.weight`), part(`${prefix}.ffn_up.weight`)]));
  packed.push(packQ40Tile32(`${prefix}.ffn_down.weight`, [part(`${prefix}.ffn_down.weight`)]));
}

const output = buildPrepackedFile(sourceSha256, packed);
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, output);
console.log(JSON.stringify({ sourcePath, outputPath, sourceSha256, tensors: packed.length, bytes: output.byteLength }, null, 2));
