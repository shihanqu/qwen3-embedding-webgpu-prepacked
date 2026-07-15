import { writeFile, mkdir } from 'node:fs/promises';
import { benchmarkTarget, printResults, type BenchmarkTarget } from './bench-lib.ts';
import { getWorkload, type WorkloadName } from './workloads.ts';

function readArgument(name: string, fallback: string): string {
  const prefix = `--${name}=`;
  return process.argv.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

const endpoint = readArgument('endpoint', 'http://127.0.0.1:1234/v1/embeddings');
const model = readArgument('model', 'text-embedding-qwen3-embedding-0.6b');
const workload = getWorkload(readArgument('workload', 'acceptance') as WorkloadName);
const durationMs = Number(readArgument('duration-ms', '10000'));
const warmupRequests = Number(readArgument('warmup', '3'));
const concurrencies = readArgument('concurrency', '1,2,4,8,16').split(',').map(Number);

const target: BenchmarkTarget = {
  name: `baseline:${model}`,
  async embed(input) {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model, input }),
    });
    if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
    const payload = await response.json() as { data?: Array<{ embedding?: number[] }> };
    const embedding = payload.data?.[0]?.embedding;
    if (!embedding) throw new Error('endpoint returned no embedding');
    return embedding;
  },
};

const results = [];
for (const concurrency of concurrencies) {
  console.log(`Benchmarking ${target.name}, workload=${workload.name}, concurrency=${concurrency} ...`);
  results.push(await benchmarkTarget(target, {
    concurrency,
    durationMs,
    warmupRequests,
    inputs: workload.inputs,
    tokensPerInput: workload.nominalTokens,
  }));
}

printResults(results);
await mkdir('bench-results', { recursive: true });
const outputPath = `bench-results/baseline-${workload.name}-${Date.now()}.json`;
await writeFile(outputPath, JSON.stringify({
  schemaVersion: 1,
  timestamp: new Date().toISOString(),
  endpoint,
  model,
  workload,
  system: { platform: process.platform, arch: process.arch },
  results,
}, null, 2));
console.log(`Saved ${outputPath}`);

