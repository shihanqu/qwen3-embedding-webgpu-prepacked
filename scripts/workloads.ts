export type WorkloadName = 'short' | 'acceptance' | 'long' | 'mixed';

export interface Workload {
  name: WorkloadName;
  nominalTokens: number;
  inputs: string[];
}

const seedSentences = [
  'A red fox crosses the quiet trail while the morning fog lifts from the valley.',
  'Vector search maps related passages nearby even when they use different vocabulary.',
  'The database transaction commits only after every invariant has been checked.',
  'WebGPU dispatches compute work to the graphics processor without a native extension.',
];

function repeatToNominalTokens(sentence: string, tokens: number): string {
  // Stable English prose averages close to 1.3 tokenizer tokens per whitespace word
  // for Qwen. The exact token count is recorded by the browser-side tokenizer later.
  const targetWords = Math.max(1, Math.floor(tokens / 1.3));
  const words = sentence.split(/\s+/);
  return Array.from({ length: targetWords }, (_, index) => words[index % words.length]).join(' ');
}

export function getWorkload(name: WorkloadName): Workload {
  const nominalTokens = name === 'short' ? 32 : name === 'long' ? 512 : 128;
  const lengths = name === 'mixed' ? [32, 64, 128, 512] : seedSentences.map(() => nominalTokens);
  return {
    name,
    nominalTokens,
    inputs: seedSentences.map((sentence, index) => repeatToNominalTokens(sentence, lengths[index])),
  };
}

