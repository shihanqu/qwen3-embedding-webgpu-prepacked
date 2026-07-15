import { describe, expect, it } from 'vitest';
import { dequantizeQ40Block, dequantizeQ4KBlock, dequantizeQ6KBlock, halfToFloat } from '../src/gguf/quantization.ts';

describe('GGML K-quant decoding', () => {
  it('decodes binary16 values', () => {
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x0000)).toBe(0);
  });

  it('decodes a Q4_K block with uniform scale and no minimum', () => {
    const block = new Uint8Array(144);
    new DataView(block.buffer).setUint16(0, 0x3c00, true); // d = 1
    block.fill(1, 4, 12); // scales/mins for subblocks 0..3
    block.fill(0x11, 12, 16); // low nibbles hold scales for subblocks 4..7
    block.fill(0x72, 16); // low nibble=2, high nibble=7
    const values = dequantizeQ4KBlock(block);
    expect(values[0]).toBe(2);
    expect(values[31]).toBe(2);
    expect(values[32]).toBe(7);
    expect(values[255]).toBe(7);
  });

  it('decodes a centered Q4_0 block', () => {
    const block = new Uint8Array(18);
    new DataView(block.buffer).setUint16(0, 0x3c00, true); // d = 1
    block.fill(0xf0, 2); // low nibble=0 -> -8, high nibble=15 -> 7
    const values = dequantizeQ40Block(block);
    expect(values[0]).toBe(-8);
    expect(values[15]).toBe(-8);
    expect(values[16]).toBe(7);
    expect(values[31]).toBe(7);
  });

  it('decodes a centered Q6_K block', () => {
    const block = new Uint8Array(210);
    block.fill(0, 0, 192); // q = 0 -> -32
    block.fill(1, 192, 208); // all scales=1
    new DataView(block.buffer).setUint16(208, 0x3c00, true); // d = 1
    const values = dequantizeQ6KBlock(block);
    expect(Array.from(values).every((value) => value === -32)).toBe(true);
  });
});
