import { describe, expect, it } from 'vitest';
import { getWorkload } from '../scripts/workloads.ts';

describe('100-token benchmark workload', () => {
  it('is explicitly labeled as an exact 100-token fixture', () => {
    const workload = getWorkload('hundred');
    expect(workload.nominalTokens).toBe(100);
    expect(workload.inputs).toHaveLength(1);
    expect(workload.inputs[0]).toContain('morning fog');
  });
});
