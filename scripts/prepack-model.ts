import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { GGUFReader, QWEN3_METADATA_KEYS } from '../src/gguf/reader.ts';
import { GGMLType, type GGUFMetadataValue } from '../src/gguf/types.ts';
import { buildPrepackedFile, packQ40Tile32, rawTensor, type PrepackedMetadataValue, type Q40TensorPart } from '../src/prepacked/format.ts';

const sourcePath = process.argv[2] ?? 'models/qwen3-embedding-0.6b-q4_0-webgpu.gguf';
const outputPath = process.argv[3] ?? 'models/qwen3-embedding-0.6b-q4_0-webgpu.wgpack';
const sourceFile = await readFile(sourcePath);
const sourceBuffer = sourceFile.buffer.slice(sourceFile.byteOffset, sourceFile.byteOffset + sourceFile.byteLength) as ArrayBuffer;
const gguf = new GGUFReader(sourceBuffer).parse({ metadataKeys: QWEN3_METADATA_KEYS });
const sourceSha256 = createHash('sha256').update(sourceFile).digest('hex');
const consumed = new Set<string>();

function jsonMetadata(value: GGUFMetadataValue): PrepackedMetadataValue {
  const convert = (item: string | number | bigint | boolean): string | number | boolean => {
    if (typeof item !== 'bigint') return item;
    const number = Number(item);
    if (!Number.isSafeInteger(number)) throw new Error(`metadata bigint ${item} is not safely representable`);
    return number;
  };
  return Array.isArray(value) ? value.map(convert) : convert(value);
}

function part(name: string): Q40TensorPart {
  const tensor = gguf.tensors.get(name);
  if (!tensor || tensor.type !== GGMLType.Q4_0 || tensor.dimensions.length !== 2) throw new Error(`${name} is not a Q4_0 matrix`);
  consumed.add(name);
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

const unexpectedQ40 = Array.from(gguf.tensors.values()).filter((tensor) => tensor.type === GGMLType.Q4_0 && tensor.dimensions.length === 2 && !consumed.has(tensor.name));
if (unexpectedQ40.length > 0) throw new Error(`unpacked Q4_0 matrices: ${unexpectedQ40.map((tensor) => tensor.name).join(', ')}`);

const raw = Array.from(gguf.tensors.values())
  .filter((tensor) => !consumed.has(tensor.name))
  .map((tensor) => rawTensor(
    tensor.name,
    tensor.dimensions,
    tensor.type,
    new Uint8Array(sourceBuffer, tensor.byteOffset, tensor.byteLength),
  ));
const metadata = Object.fromEntries(Array.from(gguf.metadata, ([key, value]) => [key, jsonMetadata(value)]));
const output = buildPrepackedFile(sourceSha256, sourceFile.byteLength, metadata, [...raw, ...packed]);
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, output);
console.log(JSON.stringify({ sourcePath, outputPath, sourceSha256, rawTensors: raw.length, packedTensors: packed.length, tensors: raw.length + packed.length, bytes: output.byteLength }, null, 2));
