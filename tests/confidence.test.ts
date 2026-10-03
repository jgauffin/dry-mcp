import { describe, it, expect } from 'vitest';
import { Clusterer } from '../src/analysis/clusterer.js';
import { hashNormalized } from '../src/chunking/normalizer.js';
import { DEFAULT_CONFIG } from '../src/config.js';
import type { Chunk, Embedding } from '../src/types.js';

/**
 * A block whose body lines are shared with every other block of the same size,
 * so the copies line up and the embedding score alone decides the verdict.
 */
function chunkOf(file: string, startLine: number, lineCount: number, marker: string): Chunk {
  const body: string[] = [`function ${marker}() {`];
  for (let i = 0; i < lineCount - 2; i++) {
    body.push(`  step${i}();`);
  }
  body.push('}');
  const text = body.join('\n');

  return {
    file,
    startLine,
    endLine: startLine + lineCount - 1,
    text,
    normalizedHash: hashNormalized(text),
    significantLines: lineCount,
  };
}

/** Two unit vectors separated by a chosen angle, so similarity is exact. */
function pairSeparatedBy(similarity: number): [Embedding, Embedding] {
  const angle = Math.acos(similarity);
  return [
    Float32Array.from([1, 0]),
    Float32Array.from([Math.cos(angle), Math.sin(angle)]),
  ];
}

/**
 * Groups two blocks that score exactly the given similarity.
 *
 * The threshold is set low enough that grouping is never what is being tested
 * here — these tests are about how a finding is judged once made, not about
 * whether it is made. Tests that do exercise the threshold set their own.
 */
function clusterWith(similarity: number, lineCount: number, threshold = 0.2) {
  const first = chunkOf('a.ts', 1, lineCount, 'alpha');
  const second = chunkOf('b.ts', 1, lineCount, 'beta');
  const [one, two] = pairSeparatedBy(similarity);

  const vectors = new Map<string, Embedding>([
    [first.normalizedHash, one],
    [second.normalizedHash, two],
  ]);

  const clusterer = new Clusterer({ similarityThreshold: threshold });
  return clusterer.cluster([first, second], vectors)[0];
}

describe('Confidence reported with each finding', () => {
  it('Code_that_is_byte_for_byte_the_same_is_reported_as_certain', () => {
    const body = ['function f() {', '  const a = 1;', '  const b = 2;', '  return a + b;', '}'].join(
      '\n'
    );
    const chunks: Chunk[] = [];
    for (const file of ['a.ts', 'b.ts']) {
      chunks.push({
        file,
        startLine: 1,
        endLine: 5,
        text: body,
        normalizedHash: hashNormalized(body),
        significantLines: 5,
      });
    }

    const [cluster] = new Clusterer({ similarityThreshold: 0.45 }).cluster(chunks, new Map());

    expect(cluster.confidence).toBe('certain');
  });

  it('A_very_close_match_over_a_normal_sized_block_is_reported_as_high', () => {
    expect(clusterWith(0.95, 20).confidence).toBe('high');
  });

  it('A_match_only_a_little_above_ordinary_is_flagged_for_reading_first', () => {
    expect(clusterWith(0.35, 15).confidence).toBe('moderate');
  });

  it('A_match_no_better_than_unrelated_code_scores_is_reported_as_low', () => {
    expect(clusterWith(0.25, 15).confidence).toBe('low');
  });

  it('The_same_score_means_less_on_a_long_block_than_on_a_short_one', () => {
    // Two hundred-line blocks of the same language reach 0.6 on syntax alone,
    // so that score is unremarkable there while it is strong evidence in a
    // twenty-line block. The long pair is not reported at all.
    expect(clusterWith(0.6, 20).confidence).toBe('high');
    expect(clusterWith(0.6, 110, DEFAULT_CONFIG.similarityThreshold)).toBeUndefined();
  });

  it('A_long_block_must_match_far_more_closely_to_earn_the_same_trust', () => {
    expect(clusterWith(0.9, 110).confidence).toBe('high');
  });

  it('A_very_short_block_is_trusted_less_because_small_code_agrees_by_chance', () => {
    const tiny = clusterWith(0.95, 7).confidence;
    const normal = clusterWith(0.95, 20).confidence;

    expect(tiny).toBe('moderate');
    expect(normal).toBe('high');
  });

  it('A_close_embedding_match_whose_lines_barely_line_up_is_not_trusted', () => {
    // Most of the body is shared, but a good part is not: the embeddings would
    // call this high on their own, the lines say it is only loosely a copy.
    const first = chunkOf('a.ts', 1, 20, 'alpha');
    const lines = first.text.split('\n');
    for (let i = 13; i < 19; i++) lines[i] = `  somethingElse${i}(withOther, args);`;
    const text = lines.join('\n');
    const second: Chunk = { ...first, file: 'b.ts', text, normalizedHash: hashNormalized(text) };

    const [one, two] = pairSeparatedBy(0.95);
    const vectors = new Map<string, Embedding>([
      [first.normalizedHash, one],
      [second.normalizedHash, two],
    ]);

    const [cluster] = new Clusterer({ similarityThreshold: 0.2 }).cluster([first, second], vectors);

    expect(cluster.alignment).toBeLessThan(0.7);
    expect(cluster.confidence).toBe('low');
  });
});

describe('Grouping long blocks', () => {
  /**
   * Long blocks of one language resemble each other on syntax alone, so a score
   * that means "the same code" for a short block means nothing for a long one.
   * Without this guard a hundred-line window matches almost any other.
   */
  const CONFIGURED = DEFAULT_CONFIG.similarityThreshold;

  it('Two_unrelated_long_blocks_are_not_grouped_at_a_score_that_would_group_short_ones', () => {
    const justAboveConfigured = CONFIGURED + 0.05;

    expect(clusterWith(justAboveConfigured, 15, CONFIGURED)).toBeDefined();
    expect(clusterWith(justAboveConfigured, 110, CONFIGURED)).toBeUndefined();
  });

  it('Two_long_blocks_that_really_are_the_same_code_are_still_grouped', () => {
    expect(clusterWith(0.9, 110, CONFIGURED)).toBeDefined();
  });
});
