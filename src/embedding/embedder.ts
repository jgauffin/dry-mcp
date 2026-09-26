import type { Embedding } from '../types.js';
import type { ModelStore } from './model-store.js';

/**
 * Turns source code into vectors whose closeness reflects how alike the code is.
 *
 * Kept behind an interface so the analysis can be exercised with known vectors,
 * without a few hundred megabytes of model and a minute of CPU per test.
 */
export interface Embedder {
  /** Vectors for a batch of code blocks, in the order they were given. */
  embed(texts: string[]): Promise<Embedding[]>;
}

/** The loaded model as Transformers.js hands it over, with the parts used here. */
export interface Extractor {
  (texts: string[], options: { pooling: string; normalize: boolean }): Promise<{
    tolist(): number[][];
  }>;
  tokenizer: {
    /** Token ids for one text, the same count the model will be given. */
    encode(text: string): number[];
    /** Where the tokenizer cuts a text off before the model sees it. */
    model_max_length: number;
  };
}

/**
 * The most blocks handed to the model at once.
 *
 * Each call carries a fixed cost that a larger batch spreads over more blocks.
 * Measured against this model, the saving has flattened out well before 32,
 * so going beyond it buys nothing and costs memory.
 */
export const MAX_BLOCKS_PER_BATCH = 32;

/**
 * The most of one block the model is asked to read, in tokens.
 *
 * The model accepts 8192, but its working memory grows with the square of the
 * length, and a single 120-line window of dense code runs to a few thousand
 * tokens. Measured on a large C# codebase, 1024 covers ninety-six blocks in a
 * hundred whole; the rest lose their tail, which two copies of the same code
 * lose identically, so they still match.
 */
export const MAX_TOKENS_PER_BLOCK = 1024;

/**
 * How much attention work one batch may ask of the model: the batch size times
 * the square of its longest block, because every block in a batch is padded to
 * the longest one.
 *
 * This is what decides the server's memory. Attention is computed for every
 * pair of tokens in every block, and the runtime keeps whatever it once needed:
 * a single batch of thirty-two long blocks took the process past ten gigabytes
 * and it never came back down. At this budget — eight blocks of 512 tokens, or
 * two at the cap — the process stays around a gigabyte on any project, and
 * indexing runs at the same speed per block as with larger batches.
 */
export const ATTENTION_BUDGET = 8 * 512 * 512;

/**
 * The real embedder, running the model locally through ONNX.
 *
 * Nothing is fetched from the network: the model is downloaded deliberately by
 * a separate command, and this refuses to reach out for it at question time.
 */
export class LocalEmbedder implements Embedder {
  private store: ModelStore;
  private pipeline: Extractor | null = null;
  private loading: Promise<Extractor> | null = null;

  constructor(store: ModelStore) {
    this.store = store;
  }

  async embed(texts: string[]): Promise<Embedding[]> {
    if (texts.length === 0) return [];

    const extractor = await this.load();

    // Counted the way the model will see them, so the budget is honoured on
    // what is actually computed rather than on a guess from character counts.
    const tokenCounts: number[] = [];
    for (const text of texts) {
      tokenCounts.push(Math.min(extractor.tokenizer.encode(text).length, MAX_TOKENS_PER_BLOCK));
    }

    const vectors: Embedding[] = new Array<Embedding>(texts.length);

    for (const batch of planBatches(tokenCounts)) {
      const batchTexts: string[] = [];
      for (const index of batch) {
        batchTexts.push(texts[index]);
      }

      // Normalizing at extraction means similarity is a plain dot product
      // later, and keeps stored vectors directly comparable.
      const output = await extractor(batchTexts, { pooling: 'mean', normalize: true });
      const rows = output.tolist();
      for (let i = 0; i < batch.length; i++) {
        vectors[batch[i]] = Float32Array.from(rows[i]);
      }
    }

    return vectors;
  }

  /**
   * Loads the model once and reuses it, since construction is the expensive
   * part. Concurrent callers share the same load rather than starting several.
   */
  private load(): Promise<Extractor> {
    if (this.pipeline) return Promise.resolve(this.pipeline);
    if (this.loading) return this.loading;

    this.loading = this.createPipeline().then((pipeline) => {
      // The tokenizer cuts every text here before the model sees it, which is
      // what keeps one oversized block from setting the cost of its whole batch.
      pipeline.tokenizer.model_max_length = MAX_TOKENS_PER_BLOCK;
      this.pipeline = pipeline;
      this.loading = null;
      return pipeline;
    });

    return this.loading;
  }

  protected async createPipeline(): Promise<Extractor> {
    const { pipeline, env } = await import('@huggingface/transformers');

    // Strictly local: a question must never silently trigger a large download.
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = this.store.rootDirectory;

    const loaded = await pipeline('feature-extraction', modelRepositoryOf(this.store), {
      dtype: this.store.precision === 'int8' ? 'q8' : 'fp32',
    });

    return loaded as unknown as Extractor;
  }
}

/**
 * Groups blocks into batches the model can hold, as lists of positions into
 * the given counts.
 *
 * Blocks are sorted by length first so that a batch is made of blocks of
 * about the same size: padding every block to the longest one in its batch
 * means a short block sharing a batch with a long one costs as much as the
 * long one. Each batch then takes blocks until adding one more would exceed
 * the attention budget or the batch size limit.
 */
export function planBatches(tokenCounts: number[]): number[][] {
  const order: number[] = [];
  for (let index = 0; index < tokenCounts.length; index++) {
    order.push(index);
  }
  order.sort((a, b) => tokenCounts[b] - tokenCounts[a]);

  const batches: number[][] = [];
  let current: number[] = [];
  let longest = 0;

  for (const index of order) {
    const length = Math.min(tokenCounts[index], MAX_TOKENS_PER_BLOCK);

    if (current.length === 0) {
      longest = length;
    }

    const wouldCost = (current.length + 1) * longest * longest;
    const full = current.length >= MAX_BLOCKS_PER_BATCH || wouldCost > ATTENTION_BUDGET;

    // A block never waits for a batch it cannot join: on its own it is always
    // affordable, because the cap bounds what a single block can cost.
    if (full && current.length > 0) {
      batches.push(current);
      current = [];
      longest = length;
    }

    current.push(index);
  }

  if (current.length > 0) batches.push(current);

  return batches;
}

function modelRepositoryOf(store: ModelStore): string {
  // The directory layout mirrors the repository name, so the tail of the model
  // directory is what Transformers.js should be asked for.
  const parts = store.modelDirectory.replace(/\\/g, '/').split('/');
  return parts.slice(-2).join('/');
}

/** Similarity of two blocks, from 0 (unrelated) to 1 (the same code). */
export function cosineSimilarity(first: Embedding, second: Embedding): number {
  if (first.length !== second.length) return 0;

  let dot = 0;
  for (let i = 0; i < first.length; i++) {
    dot += first[i] * second[i];
  }

  // Vectors are stored normalized, so the dot product is already the cosine.
  // Clamping absorbs the small excursions past 1 that float maths produces.
  if (dot > 1) return 1;
  if (dot < -1) return -1;
  return dot;
}
