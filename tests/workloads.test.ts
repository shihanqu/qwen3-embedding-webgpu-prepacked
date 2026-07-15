import { describe, expect, it } from 'vitest';
import { COMPARISON_TOKEN_COUNTS, getComparisonWorkload, getWorkload } from '../scripts/workloads.ts';

describe('100-token benchmark workload', () => {
  it('is explicitly labeled as an exact 100-token fixture', () => {
    const workload = getWorkload('hundred');
    expect(workload.nominalTokens).toBe(100);
    expect(workload.inputs).toHaveLength(1);
    expect(workload.inputs[0]).toContain('morning fog');
  });
});

describe('comparison workloads', () => {
  it('provides deterministic fixtures for every graphed token count', () => {
    expect(COMPARISON_TOKEN_COUNTS).toEqual([15, 50, 150, 500]);
    for (const tokens of COMPARISON_TOKEN_COUNTS) {
      const workload = getComparisonWorkload(tokens);
      expect(workload.nominalTokens).toBe(tokens);
      expect(workload.inputs).toHaveLength(1);
      expect(workload.inputs[0].length).toBeGreaterThan(0);
    }
  });
});
