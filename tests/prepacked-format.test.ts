import { describe, expect, it } from 'vitest';
import { buildPrepackedFile, packQ40Tile32, parsePrepackedModel } from '../src/prepacked/format.ts';

function q40Block(scaleBits: [number, number], quants: number[]): Uint8Array {
  const block = new Uint8Array(18);
  block.set(scaleBits, 0);
  for (let index = 0; index < 16; index += 1) block[index + 2] = quants[index] | (quants[index + 16] << 4);
  return block;
}

describe('prepacked WebGPU format', () => {
  it('fuses rows and stores Q4_0 blocks tile-major with word alignment', () => {
    const first = q40Block([0x00, 0x3c], Array.from({ length: 32 }, (_, index) => index & 15));
    const second = q40Block([0x00, 0x40], Array.from({ length: 32 }, (_, index) => 15 - (index & 15)));
    const packed = packQ40Tile32('fused', [
      { name: 'first', k: 32, n: 1, bytes: first },
      { name: 'second', k: 32, n: 1, bytes: second },
    ]);
    expect(packed.n).toBe(2);
    expect(packed.bytes.byteLength).toBe(32 * 32);
    expect(Array.from(packed.bytes.slice(0, 8))).toEqual([0x10, 0x32, 0x00, 0x3c, 0x54, 0x76, 0x00, 0x3c]);
    expect(Array.from(packed.bytes.slice(32, 40))).toEqual([0xef, 0xcd, 0x00, 0x40, 0xab, 0x89, 0x00, 0x40]);

    const file = buildPrepackedFile('abc123', [packed]);
    const parsed = parsePrepackedModel(file.buffer as ArrayBuffer);
    expect(parsed.header.sourceSha256).toBe('abc123');
    expect(parsed.tensors.get('fused')).toMatchObject({ k: 32, n: 2, byteLength: 1024 });
  });
});
