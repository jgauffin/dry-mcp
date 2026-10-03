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

/**
 * A block of `lineCount` lines whose body is the same for every block, with a
 * distinct first line so each block is its own shape for near-miss matching.
 */
function bodyOf(name: string, lineCount: number): string {
  const lines = [`function ${name}(order) {`];
  for (let i = 0; i < lineCount - 2; i++) {
    lines.push(`  applyDiscount${i}(order, rules);`);
  }
  lines.push('}');
  return lines.join('\n');
}

/** One window of a block too long to compare whole. */
function windowOf(
  file: string,
  startLine: number,
  text: string,
  block: { start: number; end: number }
): Chunk {
  return { ...chunkOf(file, startLine, text), blockStartLine: block.start, blockEndLine: block.end };
}

function sameDirection(chunks: Chunk[]): Map<string, Embedding> {
  const vectors = new Map<string, Embedding>();
  for (const chunk of chunks) vectors.set(chunk.normalizedHash, vectorAt(0));
  return vectors;
}

describe('Pieces of one block', () => {
  it('Two_windows_of_one_long_block_are_not_copies_however_alike_they_read', () => {
    const block = { start: 1, end: 400 };
    const chunks = [
      windowOf('src/scanner.ts', 1, bodyOf('first', 30), block),
      windowOf('src/scanner.ts', 61, bodyOf('second', 30), block),
    ];

    expect(clusterer.cluster(chunks, sameDirection(chunks))).toHaveLength(0);
  });

  it('Windows_of_two_blocks_nested_one_in_the_other_are_not_copies', () => {
    // A method opening one line inside its class: both are windowed, a line apart.
    const chunks = [
      windowOf('src/scanner.ts', 115, bodyOf('outer', 30), { start: 115, end: 400 }),
      windowOf('src/scanner.ts', 236, bodyOf('inner', 30), { start: 116, end: 399 }),
    ];

    expect(clusterer.cluster(chunks, sameDirection(chunks))).toHaveLength(0);
  });

  it('Two_separate_functions_in_one_file_can_still_be_copies', () => {
    const chunks = [
      chunkOf('src/orders.ts', 1, bodyOf('cancel', 12)),
      chunkOf('src/orders.ts', 20, bodyOf('refund', 12)),
    ];

    const clusters = clusterer.cluster(chunks, sameDirection(chunks));

    expect(clusters).toHaveLength(1);
    expect(clusters[0].frequency).toBe(2);
  });

  it('The_same_copy_found_through_windows_a_line_apart_is_reported_once', () => {
    const chunks = [
      windowOf('src/x.ts', 100, bodyOf('xOuter', 30), { start: 100, end: 300 }),
      windowOf('src/x.ts', 101, bodyOf('xInner', 30), { start: 101, end: 299 }),
      windowOf('src/y.ts', 1, bodyOf('yOuter', 30), { start: 1, end: 200 }),
      windowOf('src/y.ts', 2, bodyOf('yInner', 30), { start: 2, end: 199 }),
    ];

    const clusters = clusterer.cluster(chunks, sameDirection(chunks));

    expect(clusters).toHaveLength(1);
  });
});

describe('Code that only shares a shape', () => {
  it('Unrelated_test_suites_are_not_grouped_however_alike_their_embeddings', () => {
    const decisions = [
      "describe('decisions', () => {",
      "  it('records an accepted decision', () => {",
      "    const log = createDecisionLog();",
      "    log.accept('ship it');",
      "    expect(log.entries).toHaveLength(1);",
      '  });',
      "  it('rejects an empty rationale', () => {",
      "    expect(() => createDecisionLog().accept('')).toThrow();",
      '  });',
      '});',
    ].join('\n');
    const planView = [
      "describe('plan view', () => {",
      "  it('renders cancelled orders last', () => {",
      "    const view = renderPlan(samplePlan);",
      "    view.sortBy('status');",
      "    expect(view.rows[2].status).toBe('cancelled');",
      '  });',
      "  it('hides steps that were skipped', () => {",
      "    expect(renderPlan(skippedPlan).rows).toEqual([]);",
      '  });',
      '});',
    ].join('\n');

    const chunks = [chunkOf('test/decisions.test.ts', 1, decisions), chunkOf('test/plan-view.test.ts', 1, planView)];

    expect(clusterer.cluster(chunks, sameDirection(chunks))).toHaveLength(0);
  });

  it('A_renamed_copy_is_still_grouped_because_its_lines_still_line_up', () => {
    const original = [
      'async function cancelOrder(order, reason) {',
      '  const refund = calculateRefund(order.total, order.paidAt);',
      '  await payments.refund(order.customerId, refund);',
      "  order.status = 'cancelled';",
      '  order.cancelReason = reason;',
      '  await orders.save(order);',
      '  return refund;',
      '}',
    ].join('\n');
    const renamed = original.replace(/cancelOrder/g, 'abortPurchase').replace(/refund\b/g, 'credit');

    const chunks = [chunkOf('src/orders.ts', 1, original), chunkOf('src/purchases.ts', 1, renamed)];

    const clusters = clusterer.cluster(chunks, sameDirection(chunks));

    expect(clusters).toHaveLength(1);
    expect(clusters[0].alignment).toBeGreaterThanOrEqual(0.8);
  });
});
