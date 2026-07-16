import { GGMLType } from '../gguf/types.ts';

const MAGIC = 'WGPACK02';
const PRELUDE_BYTES = 12;
const DATA_ALIGNMENT = 256;

export const PREPACKED_LAYOUT = 'q4_0-tile32-compact-v2' as const;
export const PREPACKED_TILE_ROWS = 32;
export const PREPACKED_BLOCK_VALUES = 32;
export const PREPACKED_BLOCK_BYTES = 20;
export const PREPACKED_STORAGE_COMPACT = 'q4_0-tile32-compact' as const;
export const PREPACKED_STORAGE_RAW = 'raw' as const;

export type PrepackedMetadataScalar = string | number | boolean;
export type PrepackedMetadataValue = PrepackedMetadataScalar | PrepackedMetadataScalar[];
export type PrepackedTensorStorage = typeof PREPACKED_STORAGE_COMPACT | typeof PREPACKED_STORAGE_RAW;

export interface Q40TensorPart {
  name: string;
  k: number;
  n: number;
  bytes: Uint8Array;
}

export interface PackedTensorBytes {
  name: string;
  dimensions: number[];
  type: GGMLType;
  storage: PrepackedTensorStorage;
  bytes: Uint8Array;
}

export interface PrepackedTensorHeader {
  name: string;
  dimensions: number[];
  type: GGMLType;
  storage: PrepackedTensorStorage;
  offset: number;
  byteLength: number;
}

export interface PrepackedHeader {
  version: 2;
  sourceSha256: string;
  sourceBytes: number;
  layout: typeof PREPACKED_LAYOUT;
  tileRows: typeof PREPACKED_TILE_ROWS;
  blockValues: typeof PREPACKED_BLOCK_VALUES;
  blockBytes: typeof PREPACKED_BLOCK_BYTES;
  metadata: Record<string, PrepackedMetadataValue>;
  tensors: PrepackedTensorHeader[];
}

export interface PrepackedModel {
  header: PrepackedHeader;
  tensors: Map<string, PrepackedTensorHeader & { byteOffset: number }>;
  buffer: ArrayBuffer;
  dataOffset: number;
}

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

export function packQ40Tile32(name: string, parts: readonly Q40TensorPart[]): PackedTensorBytes {
  if (parts.length === 0) throw new Error(`cannot pack ${name} without tensor parts`);
  const k = parts[0].k;
  if (k % PREPACKED_BLOCK_VALUES !== 0) throw new Error(`${name} K=${k} is not divisible by 32`);
  for (const part of parts) {
    if (part.k !== k) throw new Error(`${name} has mismatched K dimensions`);
    const expected = part.n * (k / 32) * 18;
    if (part.bytes.byteLength !== expected) throw new Error(`${part.name} has ${part.bytes.byteLength} bytes; expected ${expected}`);
  }

  const n = parts.reduce((sum, part) => sum + part.n, 0);
  const kBlocks = k / PREPACKED_BLOCK_VALUES;
  const nTiles = Math.ceil(n / PREPACKED_TILE_ROWS);
  const output = new Uint8Array(nTiles * kBlocks * PREPACKED_TILE_ROWS * PREPACKED_BLOCK_BYTES);
  const rowParts: Array<{ part: Q40TensorPart; row: number }> = [];
  for (const part of parts) {
    for (let row = 0; row < part.n; row += 1) rowParts.push({ part, row });
  }

  for (let tile = 0; tile < nTiles; tile += 1) {
    for (let kBlock = 0; kBlock < kBlocks; kBlock += 1) {
      for (let localRow = 0; localRow < PREPACKED_TILE_ROWS; localRow += 1) {
        const outputRow = tile * PREPACKED_TILE_ROWS + localRow;
        if (outputRow >= n) continue;
        const { part, row } = rowParts[outputRow];
        const source = (row * kBlocks + kBlock) * 18;
        const destination = ((tile * kBlocks + kBlock) * PREPACKED_TILE_ROWS + localRow) * PREPACKED_BLOCK_BYTES;
        output[destination] = part.bytes[source];
        output[destination + 1] = part.bytes[source + 1];
        output.set(part.bytes.subarray(source + 2, source + 18), destination + 4);
      }
    }
  }
  return { name, dimensions: [k, n], type: GGMLType.Q4_0, storage: PREPACKED_STORAGE_COMPACT, bytes: output };
}

export function rawTensor(name: string, dimensions: number[], type: GGMLType, bytes: Uint8Array): PackedTensorBytes {
  return { name, dimensions: [...dimensions], type, storage: PREPACKED_STORAGE_RAW, bytes };
}

export function buildPrepackedFile(
  sourceSha256: string,
  sourceBytes: number,
  metadata: Record<string, PrepackedMetadataValue>,
  tensors: readonly PackedTensorBytes[],
): Uint8Array {
  let relativeOffset = 0;
  const entries: PrepackedTensorHeader[] = tensors.map((tensor) => {
    relativeOffset = align(relativeOffset, DATA_ALIGNMENT);
    const entry = {
      name: tensor.name,
      dimensions: [...tensor.dimensions],
      type: tensor.type,
      storage: tensor.storage,
      offset: relativeOffset,
      byteLength: tensor.bytes.byteLength,
    };
    relativeOffset += tensor.bytes.byteLength;
    return entry;
  });
  const header: PrepackedHeader = {
    version: 2,
    sourceSha256,
    sourceBytes,
    layout: PREPACKED_LAYOUT,
    tileRows: PREPACKED_TILE_ROWS,
    blockValues: PREPACKED_BLOCK_VALUES,
    blockBytes: PREPACKED_BLOCK_BYTES,
    metadata,
    tensors: entries,
  };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const dataOffset = align(PRELUDE_BYTES + headerBytes.byteLength, DATA_ALIGNMENT);
  const output = new Uint8Array(dataOffset + relativeOffset);
  output.set(new TextEncoder().encode(MAGIC), 0);
  new DataView(output.buffer).setUint32(8, headerBytes.byteLength, true);
  output.set(headerBytes, PRELUDE_BYTES);
  tensors.forEach((tensor, index) => output.set(tensor.bytes, dataOffset + entries[index].offset));
  return output;
}

export function parsePrepackedModel(buffer: ArrayBuffer): PrepackedModel {
  if (buffer.byteLength < PRELUDE_BYTES) throw new Error('prepacked model is shorter than its prelude');
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
  if (magic !== MAGIC) throw new Error(`invalid prepacked model magic ${JSON.stringify(magic)}`);
  const headerLength = new DataView(buffer).getUint32(8, true);
  if (PRELUDE_BYTES + headerLength > buffer.byteLength) throw new Error('prepacked model header extends past end of file');
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, PRELUDE_BYTES, headerLength))) as PrepackedHeader;
  if (header.version !== 2 || header.layout !== PREPACKED_LAYOUT || header.tileRows !== 32 || header.blockValues !== 32 || header.blockBytes !== 20) {
    throw new Error(`unsupported prepacked layout ${header.layout}`);
  }
  const dataOffset = align(PRELUDE_BYTES + headerLength, DATA_ALIGNMENT);
  const tensors = new Map<string, PrepackedTensorHeader & { byteOffset: number }>();
  for (const tensor of header.tensors) {
    const byteOffset = dataOffset + tensor.offset;
    if (byteOffset + tensor.byteLength > buffer.byteLength) throw new Error(`${tensor.name} extends past end of prepacked file`);
    if (tensors.has(tensor.name)) throw new Error(`duplicate prepacked tensor ${tensor.name}`);
    if (tensor.storage === PREPACKED_STORAGE_COMPACT && (tensor.type !== GGMLType.Q4_0 || tensor.dimensions.length !== 2)) {
      throw new Error(`${tensor.name} has invalid compact Q4_0 metadata`);
    }
    if (tensor.storage !== PREPACKED_STORAGE_COMPACT && tensor.storage !== PREPACKED_STORAGE_RAW) throw new Error(`${tensor.name} has unsupported storage ${tensor.storage}`);
    tensors.set(tensor.name, { ...tensor, byteOffset });
  }
  return { header, tensors, buffer, dataOffset };
}
