export type WorkloadName = 'tiny' | 'sentence' | 'short' | 'hundred' | 'acceptance' | 'long' | 'mixed';

export interface Workload {
  name: WorkloadName;
  nominalTokens: number;
  inputs: string[];
}

export const COMPARISON_TOKEN_COUNTS = [15, 50, 150, 500] as const;
export type ComparisonTokenCount = typeof COMPARISON_TOKEN_COUNTS[number];

const seedSentences = [
  'A red fox crosses the quiet trail while the morning fog lifts from the valley.',
  'Vector search maps related passages nearby even when they use different vocabulary.',
  'The database transaction commits only after every invariant has been checked.',
  'WebGPU dispatches compute work to the graphics processor without a native extension.',
];

const comparisonSeed = seedSentences.join(' ');
const comparisonWordCounts: Record<ComparisonTokenCount, number> = {
  15: 14,
  50: 44,
  150: 135,
  500: 446,
};

const exactHundredTokenInput = 'A red fox crosses the quiet trail while the morning fog lifts from the valley. A red fox crosses the quiet trail while the morning fog lifts from the valley. A red fox crosses the quiet trail while the morning fog lifts from the valley. A red fox crosses the quiet trail while the morning fog lifts from the valley. A red fox crosses the quiet trail while the morning fog lifts from the valley. A red fox crosses the quiet trail while the morning fog lifts from the valley. A red fox';

export function getComparisonWorkload(tokens: ComparisonTokenCount): Workload {
  const words = comparisonSeed.split(/\s+/);
  const wordCount = comparisonWordCounts[tokens];
  if (!wordCount) throw new Error(`unsupported exact-token comparison fixture: ${tokens}`);
  return {
    name: 'acceptance',
    nominalTokens: tokens,
    inputs: [Array.from({ length: wordCount }, (_, index) => words[index % words.length]).join(' ')],
  };
}

function repeatToNominalTokens(sentence: string, tokens: number): string {
  // Stable English prose averages close to 1.3 tokenizer tokens per whitespace word
  // for Qwen. The exact token count is recorded by the browser-side tokenizer later.
  const targetWords = Math.max(1, Math.floor(tokens / 1.3));
  const words = sentence.split(/\s+/);
  return Array.from({ length: targetWords }, (_, index) => words[index % words.length]).join(' ');
}

export function getWorkload(name: WorkloadName): Workload {
  if (name === 'tiny') {
    return { name, nominalTokens: 6, inputs: ['WebGPU embedding benchmark.'] };
  }
  if (name === 'sentence') {
    return { name, nominalTokens: 17, inputs: [seedSentences[0]] };
  }
  if (name === 'hundred') {
    return { name, nominalTokens: 100, inputs: [exactHundredTokenInput] };
  }
  const nominalTokens = name === 'short' ? 32 : name === 'long' ? 512 : 128;
  const lengths = name === 'mixed' ? [32, 64, 128, 512] : seedSentences.map(() => nominalTokens);
  return {
    name,
    nominalTokens,
    inputs: seedSentences.map((sentence, index) => repeatToNominalTokens(sentence, lengths[index])),
  };
}
