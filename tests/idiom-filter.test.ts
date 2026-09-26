import { describe, it, expect } from 'vitest';
import { IdiomFilter } from '../src/analysis/idiom-filter.js';
import { DEFAULT_CONFIG } from '../src/config.js';
import type { DuplicationCluster, Occurrence } from '../src/types.js';

function occurrencesIn(files: string[], lines: number): Occurrence[] {
  const result: Occurrence[] = [];
  for (const file of files) {
    result.push({ file, startLine: 1, endLine: lines, lines });
  }
  return result;
}

function cluster(overrides: Partial<DuplicationCluster>): DuplicationCluster {
  const occurrences = overrides.occurrences ?? occurrencesIn(['src/a.ts', 'src/b.ts'], 10);
  return {
    id: 'c1',
    occurrences,
    frequency: occurrences.length,
    medianLines: 10,
    removableLines: 10,
    severity: 10,
    similarity: 1,
    matchType: 'identical',
    preview: 'const value = compute();',
    ...overrides,
  };
}

describe('Idiom suppression', () => {
  const filter = new IdiomFilter(DEFAULT_CONFIG.idiom);

  it('Three_line_accessor_repeated_across_thirty_files_is_treated_as_an_idiom', () => {
    const files: string[] = [];
    for (let i = 0; i < 30; i++) {
      files.push(`src/models/entity${i}.ts`);
    }
    const accessor = cluster({
      occurrences: occurrencesIn(files, 3),
      frequency: 30,
      medianLines: 3,
    });

    const [judged] = filter.apply([accessor]);

    expect(judged.suppressed).toBeDefined();
  });

  it('Forty_line_block_in_three_files_is_always_reported_as_real_duplication', () => {
    const large = cluster({
      occurrences: occurrencesIn(['src/a.ts', 'src/b.ts', 'src/c.ts'], 40),
      frequency: 3,
      medianLines: 40,
    });

    const [judged] = filter.apply([large]);

    expect(judged.suppressed).toBeUndefined();
  });

  it('Large_block_is_reported_even_when_it_appears_very_often', () => {
    const files: string[] = [];
    for (let i = 0; i < 25; i++) {
      files.push(`src/feature${i}/handler.ts`);
    }
    const large = cluster({
      occurrences: occurrencesIn(files, 45),
      frequency: 25,
      medianLines: 45,
    });

    const [judged] = filter.apply([large]);

    expect(judged.suppressed).toBeUndefined();
  });

  it('Short_block_copied_only_twice_is_still_reported', () => {
    const pair = cluster({
      occurrences: occurrencesIn(['src/a.ts', 'src/b.ts'], 4),
      frequency: 2,
      medianLines: 4,
    });

    const [judged] = filter.apply([pair]);

    expect(judged.suppressed).toBeUndefined();
  });

  it('Duplication_confined_to_ignored_paths_is_suppressed', () => {
    const withIgnoredPaths = new IdiomFilter(DEFAULT_CONFIG.idiom, ['**/generated/**']);
    const generated = cluster({
      occurrences: occurrencesIn(['src/generated/a.ts', 'src/generated/b.ts'], 30),
      medianLines: 30,
    });

    const [judged] = withIgnoredPaths.apply([generated]);

    expect(judged.suppressed?.rule).toBe('ignored-path');
  });

  it('Duplication_spanning_ignored_and_normal_code_is_still_reported', () => {
    const withIgnoredPaths = new IdiomFilter(DEFAULT_CONFIG.idiom, ['**/generated/**']);
    const straddling = cluster({
      occurrences: occurrencesIn(['src/generated/a.ts', 'src/handwritten/b.ts'], 30),
      medianLines: 30,
    });

    const [judged] = withIgnoredPaths.apply([straddling]);

    expect(judged.suppressed).toBeUndefined();
  });

  it('Content_matching_a_configured_ignore_pattern_is_suppressed', () => {
    const withPattern = new IdiomFilter({
      ...DEFAULT_CONFIG.idiom,
      ignorePatterns: ['^\\s*public\\s+\\w+\\s+get'],
    });
    const accessor = cluster({ preview: '  public string get Name()' });

    const [judged] = withPattern.apply([accessor]);

    expect(judged.suppressed?.rule).toBe('ignored-content');
  });

  it('A_malformed_ignore_pattern_does_not_break_the_remaining_rules', () => {
    const withBadPattern = new IdiomFilter({
      ...DEFAULT_CONFIG.idiom,
      ignorePatterns: ['([unclosed'],
    });
    const normal = cluster({});

    expect(() => withBadPattern.apply([normal])).not.toThrow();
  });

  it('Suppressed_clusters_explain_themselves_so_the_rules_can_be_tuned', () => {
    const files: string[] = [];
    for (let i = 0; i < 30; i++) {
      files.push(`src/models/entity${i}.ts`);
    }
    const accessor = cluster({
      occurrences: occurrencesIn(files, 3),
      frequency: 30,
      medianLines: 3,
    });

    const [judged] = filter.apply([accessor]);

    expect(judged.suppressed!.explanation.length).toBeGreaterThan(0);
  });
});
