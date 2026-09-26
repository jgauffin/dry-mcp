import { describe, it, expect } from 'vitest';
import {
  severityOf,
  removableLines,
  medianLinesOf,
  rankBySeverity,
  rankByFrequency,
} from '../src/analysis/ranker.js';
import type { DuplicationCluster } from '../src/types.js';

function cluster(id: string, medianLines: number, frequency: number): DuplicationCluster {
  return {
    id,
    occurrences: [],
    frequency,
    medianLines,
    removableLines: removableLines(medianLines, frequency),
    severity: severityOf(medianLines, frequency),
    similarity: 1,
    matchType: 'identical',
    preview: id,
  };
}

describe('Duplication ranking', () => {
  it('Large_clone_repeated_four_times_outranks_tiny_fragment_repeated_forty_times', () => {
    const large = cluster('large', 60, 4);
    const tiny = cluster('tiny', 3, 40);

    const ranked = rankBySeverity([tiny, large]);

    expect(ranked[0].id).toBe('large');
  });

  it('Ordering_by_frequency_puts_the_most_widespread_copy_first', () => {
    const large = cluster('large', 60, 4);
    const tiny = cluster('tiny', 3, 40);

    const ranked = rankByFrequency([large, tiny]);

    expect(ranked[0].id).toBe('tiny');
  });

  it('Only_the_copies_beyond_the_first_count_as_waste', () => {
    // Four copies of ten lines means thirty lines could be deleted, not forty.
    expect(removableLines(10, 4)).toBe(30);
  });

  it('Code_appearing_once_is_not_duplication_and_scores_nothing', () => {
    expect(severityOf(100, 1)).toBe(0);
  });

  it('Between_equal_sized_blocks_the_more_frequent_one_ranks_higher', () => {
    const rare = cluster('rare', 20, 2);
    const common = cluster('common', 20, 8);

    const ranked = rankBySeverity([rare, common]);

    expect(ranked[0].id).toBe('common');
  });

  it('Between_equally_frequent_blocks_the_larger_one_ranks_higher', () => {
    const small = cluster('small', 8, 3);
    const big = cluster('big', 40, 3);

    const ranked = rankBySeverity([small, big]);

    expect(ranked[0].id).toBe('big');
  });

  it('Typical_copy_size_ignores_a_single_unusually_long_occurrence', () => {
    const occurrences = [
      { file: 'a', startLine: 1, endLine: 10, lines: 10 },
      { file: 'b', startLine: 1, endLine: 10, lines: 10 },
      { file: 'c', startLine: 1, endLine: 400, lines: 400 },
    ];

    expect(medianLinesOf(occurrences)).toBe(10);
  });
});
