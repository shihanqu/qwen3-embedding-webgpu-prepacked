import { describe, expect, it } from 'vitest';
import { GGMLType } from '../src/gguf/types.ts';
import { buildPrepackedFile, packQ40Tile32, parsePrepackedModel, rawTensor } from '../src/prepacked/format.ts';

function q40Block(scaleBits: [number, number], quants: number[]): Uint8Array {
  const block = new Uint8Array(18);
  block.set(scaleBits, 0);
  for (let index = 0; index < 16; index += 1) block[index + 2] = quants[index] | (quants[index + 16] << 4);
  return block;
}

describe('prepacked WebGPU format', () => {
  it('builds a self-contained file with compact tile-major Q4_0 and raw tensors', () => {
    const first = q40Block([0x00, 0x3c], Array.from({ length: 32 }, (_, index) => index & 15));
    const second = q40Block([0x00, 0x40], Array.from({ length: 32 }, (_, index) => 15 - (index & 15)));
    const packed = packQ40Tile32('fused', [
      { name: 'first', k: 32, n: 1, bytes: first },
      { name: 'second', k: 32, n: 1, bytes: second },
    ]);
    expect(packed.dimensions).toEqual([32, 2]);
    expect(packed.bytes.byteLength).toBe(32 * 20);
    expect(Array.from(packed.bytes.slice(0, 8))).toEqual([0x00, 0x3c, 0x00, 0x00, 0x00, 0x11, 0x22, 0x33]);
    expect(Array.from(packed.bytes.slice(20, 28))).toEqual([0x00, 0x40, 0x00, 0x00, 0xff, 0xee, 0xdd, 0xcc]);

    const raw = rawTensor('norm', [1], GGMLType.F32, new Uint8Array([1, 2, 3, 4]));
    const file = buildPrepackedFile('abc123', 42, { 'general.name': 'test' }, [raw, packed]);
    const parsed = parsePrepackedModel(file.buffer as ArrayBuffer);
    expect(parsed.header.sourceSha256).toBe('abc123');
    expect(parsed.header.sourceBytes).toBe(42);
    expect(parsed.header.metadata['general.name']).toBe('test');
    expect(parsed.tensors.get('norm')).toMatchObject({ dimensions: [1], type: GGMLType.F32, storage: 'raw', byteLength: 4 });
    expect(parsed.tensors.get('fused')).toMatchObject({ dimensions: [32, 2], type: GGMLType.Q4_0, storage: 'q4_0-tile32-compact', byteLength: 640 });
  });
});
