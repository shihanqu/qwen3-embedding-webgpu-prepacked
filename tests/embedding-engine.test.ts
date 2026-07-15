import { describe, expect, it } from 'vitest';
import { Qwen3EmbeddingEngine, type EmbeddingPlan, type EmbeddingRuntime } from '../src/webgpu/embedding-engine.ts';

describe('Qwen3EmbeddingEngine', () => {
  it('coalesces up to 16 simultaneous requests and preserves result order', async () => {
    const calls: Array<{ batch: number; sequence: number; ids: number[]; lengths: number[] }> = [];
    const runtime: EmbeddingRuntime = {
      createPlan(batch, sequence): EmbeddingPlan {
        return {
          async run(ids, lengths) {
            calls.push({ batch, sequence, ids: Array.from(ids), lengths: Array.from(lengths) });
            return Array.from({ length: batch }, (_, index) => new Float32Array([ids[index * sequence]]));
          },
        };
      },
    };
    const tokenizer = (text: string) => ({ input_ids: { tolist: () => [[Number(text)]] } });
    const engine = new Qwen3EmbeddingEngine(runtime, tokenizer, 16);
    const results = await Promise.all(Array.from({ length: 16 }, (_, index) => engine.embed(String(index + 1))));
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ batch: 16, sequence: 2, lengths: Array(16).fill(2) });
    expect(results.map((value) => value[0])).toEqual(Array.from({ length: 16 }, (_, index) => index + 1));
  });

  it('pads mixed-length requests after their final token', async () => {
    let recorded: number[] = [];
    const runtime: EmbeddingRuntime = {
      createPlan(batch, sequence) {
        return { async run(ids) { recorded = Array.from(ids); return Array.from({ length: batch }, () => new Float32Array(1)); } };
      },
    };
    const tokenizer = (text: string) => ({ input_ids: { tolist: () => [text.split(',').map(Number)] } });
    const engine = new Qwen3EmbeddingEngine(runtime, tokenizer, 16, 0, 99);
    await Promise.all([engine.embed('1'), engine.embed('2,3')]);
    expect(recorded).toEqual([1, 99, 99, 2, 3, 99]);
  });
});
