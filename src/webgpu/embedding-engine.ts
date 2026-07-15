export interface TokenizerResult {
  input_ids: { tolist(): number[][] };
}

export type TokenizerLike = (text: string) => TokenizerResult;

export interface EmbeddingPlan {
  run(ids: Uint32Array, lengths: Uint32Array): Promise<Float32Array[]>;
}

export interface EmbeddingRuntime {
  createPlan(batch: number, sequence: number): EmbeddingPlan;
}

interface PendingRequest {
  tokens: number[];
  resolve(value: Float32Array): void;
  reject(reason: unknown): void;
}

/** Micro-batches simultaneous embedding calls onto one WebGPU execution plan. */
export class Qwen3EmbeddingEngine {
  private readonly pending: PendingRequest[] = [];
  private readonly plans = new Map<string, EmbeddingPlan>();
  private scheduled = false;
  private flushing = false;

  constructor(
    private readonly runtime: EmbeddingRuntime,
    private readonly tokenizer: TokenizerLike,
    readonly maxBatchSize = 16,
    readonly batchWindowMs = 0,
    readonly eosToken = 151643,
  ) {
    if (maxBatchSize < 1 || maxBatchSize > 16) throw new Error('maxBatchSize must be between 1 and 16');
  }

  embed(text: string): Promise<Float32Array> {
    const tokens = this.tokenizer(text).input_ids.tolist()[0].map(Number);
    if (tokens[tokens.length - 1] !== this.eosToken) tokens.push(this.eosToken);
    return new Promise((resolve, reject) => {
      this.pending.push({ tokens, resolve, reject });
      this.schedule();
    });
  }

  private schedule(): void {
    if (this.scheduled || this.flushing) return;
    this.scheduled = true;
    setTimeout(() => {
      this.scheduled = false;
      void this.flush();
    }, this.batchWindowMs);
  }

  private async flush(): Promise<void> {
    if (this.flushing) return;
    this.flushing = true;
    try {
      while (this.pending.length > 0) {
        const requests = this.pending.splice(0, this.maxBatchSize);
        const sequence = Math.max(...requests.map((request) => request.tokens.length));
        const ids = new Uint32Array(requests.length * sequence);
        ids.fill(this.eosToken);
        const lengths = new Uint32Array(requests.length);
        for (let index = 0; index < requests.length; index += 1) {
          ids.set(requests[index].tokens, index * sequence);
          lengths[index] = requests[index].tokens.length;
        }
        const key = `${requests.length}:${sequence}`;
        let plan = this.plans.get(key);
        if (!plan) {
          plan = this.runtime.createPlan(requests.length, sequence);
          this.plans.set(key, plan);
        }
        try {
          const embeddings = await plan.run(ids, lengths);
          requests.forEach((request, index) => request.resolve(embeddings[index]));
        } catch (error) {
          requests.forEach((request) => request.reject(error));
        }
      }
    } finally {
      this.flushing = false;
      if (this.pending.length > 0) this.schedule();
    }
  }
}
