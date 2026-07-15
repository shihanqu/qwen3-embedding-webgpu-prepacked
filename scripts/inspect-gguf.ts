import { readFile } from 'node:fs/promises';
import { GGUFReader, QWEN3_METADATA_KEYS } from '../src/gguf/reader.ts';

const path = process.argv[2] ?? 'models/qwen3-embedding-0.6b-q4_0-webgpu.gguf';
const file = await readFile(path);
const buffer = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength) as ArrayBuffer;
const model = new GGUFReader(buffer).parse({ metadataKeys: QWEN3_METADATA_KEYS });
console.log(Object.fromEntries(model.metadata));
console.table(Array.from(model.tensors.values()).slice(0, 16).map((tensor) => ({
  name: tensor.name,
  dimensions: tensor.dimensions.join(' × '),
  type: tensor.type,
  MiB: (tensor.byteLength / 2 ** 20).toFixed(2),
  byteOffset: tensor.byteOffset,
})));
console.log(`${model.tensors.size} tensors; data starts at ${model.dataOffset}; file ${(buffer.byteLength / 2 ** 20).toFixed(1)} MiB`);
