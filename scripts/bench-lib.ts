import { performance } from 'node:perf_hooks';

export interface BenchmarkTarget {
  name: string;
  embed(input: string): Promise<number[]>;
}

export interface BenchmarkOptions {
  concurrency: number;
  durationMs: number;
  warmupRequests: number;
  inputs: readonly string[];
  tokensPerInput: number;
}

export interface BenchmarkResult {
  target: string;
  concurrency: number;
  requests: number;
  errors: number;
  elapsedMs: number;
  requestsPerSecond: number;
  tokensPerSecond: number;
  latencyMs: { p50: number; p95: number; p99: number; mean: number };
  embeddingDimensions: number;
}

export function percentile(sorted: readonly number[], fraction: number): number {
  if (sorted.length === 0) return Number.NaN;
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1);
  return sorted[index];
}

export async function benchmarkTarget(
  target: BenchmarkTarget,
  options: BenchmarkOptions,
): Promise<BenchmarkResult> {
  if (options.concurrency < 1) throw new Error('concurrency must be >= 1');
  if (options.inputs.length === 0) throw new Error('at least one input is required');

  for (let i = 0; i < options.warmupRequests; i += 1) {
    await target.embed(options.inputs[i % options.inputs.length]);
  }

  const deadline = performance.now() + options.durationMs;
  const latencies: number[] = [];
  let requests = 0;
  let errors = 0;
  let embeddingDimensions = 0;
  let nextInput = 0;
  const startedAt = performance.now();

  async function worker(): Promise<void> {
    while (performance.now() < deadline) {
      const input = options.inputs[nextInput++ % options.inputs.length];
      const requestStarted = performance.now();
      try {
        const embedding = await target.embed(input);
        embeddingDimensions ||= embedding.length;
        if (embedding.length !== embeddingDimensions) {
          throw new Error(`embedding dimension changed to ${embedding.length}`);
        }
        requests += 1;
      } catch (error) {
        errors += 1;
        if (errors === 1) console.error(error);
      } finally {
        latencies.push(performance.now() - requestStarted);
      }
    }
  }

  await Promise.all(Array.from({ length: options.concurrency }, worker));
  const elapsedMs = performance.now() - startedAt;
  latencies.sort((a, b) => a - b);
  const successfulLatencies = latencies.slice(0, requests);
  const mean = successfulLatencies.reduce((sum, value) => sum + value, 0) / requests;
  const requestsPerSecond = requests / (elapsedMs / 1_000);

  return {
    target: target.name,
    concurrency: options.concurrency,
    requests,
    errors,
    elapsedMs,
    requestsPerSecond,
    tokensPerSecond: requestsPerSecond * options.tokensPerInput,
    latencyMs: {
      p50: percentile(successfulLatencies, 0.5),
      p95: percentile(successfulLatencies, 0.95),
      p99: percentile(successfulLatencies, 0.99),
      mean,
    },
    embeddingDimensions,
  };
}

export function printResults(results: readonly BenchmarkResult[]): void {
  const one = results.find((result) => result.concurrency === 1);
  console.table(results.map((result) => ({
    target: result.target,
    concurrency: result.concurrency,
    requests: result.requests,
    errors: result.errors,
    'req/s': result.requestsPerSecond.toFixed(2),
    'tok/s': result.tokensPerSecond.toFixed(0),
    'p50 ms': result.latencyMs.p50.toFixed(1),
    'p95 ms': result.latencyMs.p95.toFixed(1),
    scaling: one ? `${(result.requestsPerSecond / one.requestsPerSecond).toFixed(2)}x` : 'n/a',
  })));
}

