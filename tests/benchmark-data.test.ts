import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

interface BenchmarkRow {
  tokens: number;
  webgpu: {
    singleRps: number;
    concurrency16AggregateRps: number;
    lmStudioCosine: number;
    worstBatchCosine: number;
  };
  lmStudio: { singleRps: number; concurrency16AggregateRps: number };
}

const data = JSON.parse(readFileSync(resolve('docs/benchmarks/2026-07-16-webgpu-vs-lm-studio-m3-max.json'), 'utf8')) as { rows: BenchmarkRow[] };

describe('published benchmark matrix', () => {
  it('contains every required exact-token workload', () => {
    expect(data.rows.map((row) => row.tokens)).toEqual([15, 50, 150, 500, 1500, 5000]);
  });

  it.each(data.rows)('$tokens tokens beats raw LM Studio and clears accuracy gates', (row) => {
    expect(row.webgpu.singleRps).toBeGreaterThan(row.lmStudio.singleRps);
    expect(row.webgpu.concurrency16AggregateRps).toBeGreaterThan(row.lmStudio.concurrency16AggregateRps);
    expect(row.webgpu.lmStudioCosine).toBeGreaterThanOrEqual(0.90);
    expect(row.webgpu.worstBatchCosine).toBeGreaterThanOrEqual(0.999);
  });
});
