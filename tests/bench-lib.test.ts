import { describe, expect, it } from 'vitest';
import { benchmarkTarget, percentile } from '../scripts/bench-lib.ts';

describe('benchmark utilities', () => {
  it('computes nearest-rank percentiles', () => {
    expect(percentile([1, 2, 3, 4], 0.5)).toBe(2);
    expect(percentile([1, 2, 3, 4], 0.95)).toBe(4);
  });

  it('runs the requested number of concurrent workers', async () => {
    let active = 0;
    let peak = 0;
    const result = await benchmarkTarget({
      name: 'fake',
      async embed() {
        active += 1;
        peak = Math.max(peak, active);
        await new Promise((resolve) => setTimeout(resolve, 2));
        active -= 1;
        return [1, 2, 3];
      },
    }, {
      concurrency: 4,
      durationMs: 20,
      warmupRequests: 0,
      inputs: ['input'],
      tokensPerInput: 1,
    });

    expect(peak).toBe(4);
    expect(result.requests).toBeGreaterThan(0);
    expect(result.embeddingDimensions).toBe(3);
  });
});

