import { describe, it, expect } from 'vitest';
import { Clusterer } from '../src/analysis/clusterer.js';
import { hashNormalized } from '../src/chunking/normalizer.js';
import type { Chunk, Embedding } from '../src/types.js';

function chunkOf(file: string, startLine: number, text: string): Chunk {
  const lines = text.split('\n');
  return {
    file,
    startLine,
    endLine: startLine + lines.length - 1,
    text,
    normalizedHash: hashNormalized(text),
    significantLines: lines.length,
  };
}

/** A vector pointing in a chosen direction, so similarity is predictable. */
function vectorAt(angleInDegrees: number): Embedding {
  const radians = (angleInDegrees * Math.PI) / 180;
  return Float32Array.from([Math.cos(radians), Math.sin(radians)]);
}

const clusterer = new Clusterer({ similarityThreshold: 0.92 });

describe('Duplication clustering', () => {
  it('The_same_code_in_two_files_is_reported_as_one_duplication', () => {
    const body = ['function f() {', '  const a = 1;', '  return a;', '}'].join('\n');
    const chunks = [chunkOf('src/a.ts', 1, body), chunkOf('src/b.ts', 10, body)];

    const clusters = clusterer.cluster(chunks, new Map());

    expect(clusters).toHaveLength(1);
    expect(clusters[0].frequency).toBe(2);
    expect(clusters[0].matchType).toBe('identical');
  });

  it('Code_appearing_once_is_not_reported', () => {
    const chunks = [chunkOf('src/a.ts', 1, 'function f() {\n  return 1;\n}')];

    expect(clusterer.cluster(chunks, new Map())).toHaveLength(0);
  });

  it('Identical_copies_are_found_even_when_no_vectors_are_available_yet', () => {
    const body = ['function f() {', '  const a = 1;', '  return a;', '}'].join('\n');
    const chunks = [chunkOf('src/a.ts', 1, body), chunkOf('src/b.ts', 1, body)];

    // An empty vector map stands for a project still waiting in the queue.
    const clusters = clusterer.cluster(chunks, new Map());

    expect(clusters).toHaveLength(1);
  });

  it('Renamed_copies_are_grouped_as_near_identical_when_their_meaning_matches', () => {
    const original = ['function total(order) {', '  let sum = 0;', '  return sum;', '}'].join('\n');
    const renamed = ['function total(basket) {', '  let acc = 0;', '  return acc;', '}'].join('\n');

    const first = chunkOf('src/a.ts', 1, original);
    const second = chunkOf('src/b.ts', 1, renamed);

    const vectors = new Map<string, Embedding>([
      [first.normalizedHash, vectorAt(0)],
      [second.normalizedHash, vectorAt(10)],
    ]);

    const clusters = clusterer.cluster([first, second], vectors);

    expect(clusters).toHaveLength(1);
    expect(clusters[0].matchType).toBe('near-identical');
  });

  it('Unrelated_code_is_not_grouped_however_close_the_blocks_sit', () => {
    const first = chunkOf('src/a.ts', 1, 'function parse(x) {\n  return JSON.parse(x);\n}');
    const second = chunkOf('src/b.ts', 1, 'function render(y) {\n  return draw(y);\n}');

    const vectors = new Map<string, Embedding>([
      [first.normalizedHash, vectorAt(0)],
      [second.normalizedHash, vectorAt(80)],
    ]);

    expect(clusterer.cluster([first, second], vectors)).toHaveLength(0);
  });

  it('Blocks_of_very_different_length_are_never_called_the_same_code', () => {
    const short = chunkOf('src/a.ts', 1, ['a();', 'b();', 'c();'].join('\n'));
    const longLines: string[] = [];
    for (let i = 0; i < 30; i++) {
      longLines.push(`step${i}();`);
    }
    const long = chunkOf('src/b.ts', 1, longLines.join('\n'));

    const vectors = new Map<string, Embedding>([
      [short.normalizedHash, vectorAt(0)],
      [long.normalizedHash, vectorAt(0)],
    ]);

    expect(clusterer.cluster([short, long], vectors)).toHaveLength(0);
  });

  it('A_block_and_the_window_inside_it_are_not_reported_as_copies_of_each_other', () => {
    const body = ['a();', 'b();', 'c();'].join('\n');
    const outer = chunkOf('src/a.ts', 1, body);
    const inner = { ...chunkOf('src/a.ts', 1, body), endLine: 2 };

    const clusters = clusterer.cluster([outer, inner], new Map());

    expect(clusters).toHaveLength(0);
  });

  it('Every_place_the_code_appears_is_listed_so_it_can_be_acted_on', () => {
    const body = ['function f() {', '  const a = 1;', '  return a;', '}'].join('\n');
    const chunks = [
      chunkOf('src/b.ts', 20, body),
      chunkOf('src/a.ts', 5, body),
      chunkOf('src/c.ts', 1, body),
    ];

    const [cluster] = clusterer.cluster(chunks, new Map());

    expect(cluster.occurrences.map((o) => o.file)).toEqual(['src/a.ts', 'src/b.ts', 'src/c.ts']);
    expect(cluster.occurrences[0].startLine).toBe(5);
  });

  it('A_cluster_keeps_the_same_identity_so_its_source_can_be_fetched_later', () => {
    const body = ['function f() {', '  const a = 1;', '  return a;', '}'].join('\n');
    const chunks = [chunkOf('src/a.ts', 1, body), chunkOf('src/b.ts', 1, body)];

    const first = clusterer.cluster(chunks, new Map())[0];
    const second = clusterer.cluster(chunks, new Map())[0];

    expect(first.id).toBe(second.id);
  });

  it('Three_copies_of_the_same_block_form_a_single_group_not_three_pairs', () => {
    const body = ['function f() {', '  const a = 1;', '  return a;', '}'].join('\n');
    const chunks = [
      chunkOf('src/a.ts', 1, body),
      chunkOf('src/b.ts', 1, body),
      chunkOf('src/c.ts', 1, body),
    ];

    const clusters = clusterer.cluster(chunks, new Map());

    expect(clusters).toHaveLength(1);
    expect(clusters[0].frequency).toBe(3);
  });
});
