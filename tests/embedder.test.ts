import { describe, it, expect } from 'vitest';
import {
  LocalEmbedder,
  planBatches,
  MAX_BLOCKS_PER_BATCH,
  MAX_TOKENS_PER_BLOCK,
  ATTENTION_BUDGET,
  type Extractor,
} from '../src/embedding/embedder.js';
import { ModelStore } from '../src/embedding/model-store.js';
import type { Embedding } from '../src/types.js';

/**
 * The model's working memory for one batch, in attention cells. Costed at the
 * cap, because that is all of a block the model is ever shown.
 */
function attentionWorkOf(tokenCounts: number[], batch: number[]): number {
  let longest = 0;
  for (const index of batch) {
    const seen = Math.min(tokenCounts[index], MAX_TOKENS_PER_BLOCK);
    if (seen > longest) longest = seen;
  }
  return batch.length * longest * longest;
}

function repeated(value: number, times: number): number[] {
  const values: number[] = [];
  for (let i = 0; i < times; i++) values.push(value);
  return values;
}

describe('How blocks are batched for the model', () => {
  it('Short_blocks_fill_a_batch_so_indexing_stays_fast', () => {
    const batches = planBatches(repeated(120, 100));

    expect(batches[0].length).toBe(MAX_BLOCKS_PER_BATCH);
  });

  it('No_batch_asks_the_model_for_more_attention_work_than_the_budget', () => {
    const counts = repeated(1400, 32);

    for (const batch of planBatches(counts)) {
      expect(attentionWorkOf(counts, batch)).toBeLessThanOrEqual(ATTENTION_BUDGET);
    }
  });

  it('Long_blocks_go_alone_rather_than_being_dropped', () => {
    const batches = planBatches([MAX_TOKENS_PER_BLOCK, 50, MAX_TOKENS_PER_BLOCK]);

    let covered = 0;
    for (const batch of batches) covered += batch.length;
    expect(covered).toBe(3);
  });

  it('A_block_longer_than_the_cap_is_costed_at_the_cap_because_the_model_only_sees_that_much', () => {
    const batches = planBatches([MAX_TOKENS_PER_BLOCK * 4]);

    expect(batches).toEqual([[0]]);
  });

  it('Blocks_are_handed_over_longest_first_so_short_ones_mostly_batch_with_their_peers', () => {
    const counts = [1400, 100, 1400, 100, 1400, 100];
    const handedOver: number[] = [];

    for (const batch of planBatches(counts)) {
      for (const index of batch) handedOver.push(counts[index]);
    }

    expect(handedOver).toEqual([1400, 1400, 1400, 100, 100, 100]);
  });

  it('Every_block_is_planned_exactly_once', () => {
    const counts = [1400, 100, 900, 2048, 30, 30, 700];
    const seen: number[] = [];

    for (const batch of planBatches(counts)) {
      for (const index of batch) seen.push(index);
    }

    seen.sort((a, b) => a - b);
    expect(seen).toEqual([0, 1, 2, 3, 4, 5, 6]);
  });
});

/**
 * Stands in for the model: a "vector" is the block's own length, so the caller
 * can tell which vector belongs to which block, and every batch handed over is
 * recorded so its shape can be checked.
 */
class RecordingExtractor {
  batches: string[][] = [];
  tokenizer = {
    model_max_length: 8192,
    encode(text: string): number[] {
      return new Array<number>(Math.ceil(text.length / 4)).fill(1);
    },
  };

  async call(texts: string[]): Promise<{ tolist(): number[][] }> {
    this.batches.push(texts);
    const rows: number[][] = [];
    for (const text of texts) rows.push([text.length]);
    return { tolist: () => rows };
  }
}

function extractorFrom(recording: RecordingExtractor): Extractor {
  const extractor = ((texts: string[]) => recording.call(texts)) as Extractor;
  extractor.tokenizer = recording.tokenizer;
  return extractor;
}

class FakeModelEmbedder extends LocalEmbedder {
  constructor(private readonly recording: RecordingExtractor) {
    super(new ModelStore({ precision: 'int8', path: null }));
  }

  protected override async createPipeline(): Promise<Extractor> {
    return extractorFrom(this.recording);
  }
}

describe('Embedding through the local model', () => {
  it('Vectors_come_back_in_the_order_the_blocks_were_given_even_when_batches_are_reordered', async () => {
    const recording = new RecordingExtractor();
    const embedder = new FakeModelEmbedder(recording);
    const texts = ['x'.repeat(6000), 'y'.repeat(40), 'z'.repeat(6000), 'w'.repeat(40)];

    const vectors: Embedding[] = await embedder.embed(texts);

    expect(Array.from(vectors[0])).toEqual([6000]);
    expect(Array.from(vectors[1])).toEqual([40]);
    expect(Array.from(vectors[2])).toEqual([6000]);
    expect(Array.from(vectors[3])).toEqual([40]);
  });

  it('The_model_is_told_to_read_no_more_of_a_block_than_the_cap', async () => {
    const recording = new RecordingExtractor();
    const embedder = new FakeModelEmbedder(recording);

    await embedder.embed(['a'.repeat(100)]);

    expect(recording.tokenizer.model_max_length).toBe(MAX_TOKENS_PER_BLOCK);
  });

  it('Long_blocks_are_not_handed_over_thirty_two_at_a_time', async () => {
    const recording = new RecordingExtractor();
    const embedder = new FakeModelEmbedder(recording);
    const texts: string[] = [];
    for (let i = 0; i < 32; i++) texts.push('c'.repeat(1400 * 4));

    await embedder.embed(texts);

    const seen = Math.min(1400, MAX_TOKENS_PER_BLOCK);
    for (const batch of recording.batches) {
      expect(batch.length * seen * seen).toBeLessThanOrEqual(ATTENTION_BUDGET);
    }
  });
});
