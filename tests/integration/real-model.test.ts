import { describe, it, expect, beforeAll } from 'vitest';
import { LocalEmbedder, cosineSimilarity } from '../../src/embedding/embedder.js';
import { ModelStore } from '../../src/embedding/model-store.js';
import { DEFAULT_CONFIG } from '../../src/config.js';
import { MIN_ALIGNMENT } from '../../src/analysis/clusterer.js';
import { alignment, contentLines } from '../../src/analysis/line-alignment.js';

/**
 * Exercises the real embedding model, which has to be downloaded first:
 *
 *   npx duplication-mcp download-model
 *
 * Kept out of the default test run so ordinary development stays offline and
 * fast. What is proved here cannot be proved with a stand-in: that renamed code
 * really does land close together and unrelated code really does not.
 */
const ORIGINAL = [
  'function calculateOrderTotal(order) {',
  '  let total = 0;',
  '  for (const line of order.lines) {',
  '    total += line.quantity * line.unitPrice;',
  '  }',
  '  return total;',
  '}',
].join('\n');

/** The same logic with every name changed. */
const RENAMED = [
  'function computeBasketSum(basket) {',
  '  let sum = 0;',
  '  for (const item of basket.items) {',
  '    sum += item.count * item.price;',
  '  }',
  '  return sum;',
  '}',
].join('\n');

/** Different logic entirely. */
const UNRELATED = [
  'function parseConnectionString(raw) {',
  '  const parts = raw.split(";");',
  '  const settings = new Map();',
  '  for (const part of parts) {',
  '    const [key, value] = part.split("=");',
  '    settings.set(key.trim(), value?.trim());',
  '  }',
  '  return settings;',
  '}',
].join('\n');

/**
 * The hardest thing to get right: the same shape as the original — accumulate
 * over a collection in a loop, return the accumulator — doing unrelated work.
 * A detector that matches on shape alone reports this, and is wrong.
 */
const SAME_SHAPE_DIFFERENT_WORK = [
  'function countErrorsBySeverity(report) {',
  '  let critical = 0;',
  '  for (const finding of report.findings) {',
  '    critical += finding.severity === "high" ? 1 : 0;',
  '  }',
  '  return critical;',
  '}',
].join('\n');

/** The same logic a developer rewrote in another language. */
const SAME_LOGIC_IN_CSHARP = [
  'public decimal CalculateOrderTotal(Order order) {',
  '    decimal total = 0;',
  '    foreach (var line in order.Lines) {',
  '        total += line.Quantity * line.UnitPrice;',
  '    }',
  '    return total;',
  '}',
].join('\n');

describe('The real embedding model', () => {
  const store = new ModelStore(DEFAULT_CONFIG.model);

  beforeAll(() => {
    if (store.status() !== 'ready') {
      throw new Error(
        `These tests need the model. ${store.explainUnavailable()}`
      );
    }
  });

  it('Code_that_differs_only_in_naming_is_recognised_as_the_same_logic', async () => {
    const embedder = new LocalEmbedder(store);

    const [original, renamed] = await embedder.embed([ORIGINAL, RENAMED]);
    const similarity = cosineSimilarity(original, renamed);

    expect(similarity).toBeGreaterThan(DEFAULT_CONFIG.similarityThreshold);
  });

  it('Code_doing_something_different_is_not_treated_as_a_copy', async () => {
    const embedder = new LocalEmbedder(store);

    const [original, unrelated] = await embedder.embed([ORIGINAL, UNRELATED]);
    const similarity = cosineSimilarity(original, unrelated);

    expect(similarity).toBeLessThan(DEFAULT_CONFIG.similarityThreshold);
  });

  it('A_loop_of_the_same_shape_doing_unrelated_work_is_not_a_copy', async () => {
    const embedder = new LocalEmbedder(store);

    const [original, sameShape] = await embedder.embed([ORIGINAL, SAME_SHAPE_DIFFERENT_WORK]);
    const similarity = cosineSimilarity(original, sameShape);

    // The hardest negative: accumulating over a collection in a loop, exactly
    // like the original, but counting something else entirely. If the threshold
    // ever drifts below this, the report fills with false findings.
    expect(similarity).toBeLessThan(DEFAULT_CONFIG.similarityThreshold);
  });

  it('The_threshold_sits_between_real_copies_and_unrelated_code', async () => {
    const embedder = new LocalEmbedder(store);

    const [original, renamed, unrelated, sameShape] = await embedder.embed([
      ORIGINAL,
      RENAMED,
      UNRELATED,
      SAME_SHAPE_DIFFERENT_WORK,
    ]);

    const lowestCopy = cosineSimilarity(original, renamed);
    const highestNonCopy = Math.max(
      cosineSimilarity(original, unrelated),
      cosineSimilarity(original, sameShape)
    );

    // The gap is what makes the setting meaningful at all. Guarding it means a
    // model change that collapses the two groups fails loudly here rather than
    // quietly producing nonsense findings.
    expect(lowestCopy).toBeGreaterThan(highestNonCopy);
    expect(DEFAULT_CONFIG.similarityThreshold).toBeGreaterThan(highestNonCopy);
    expect(DEFAULT_CONFIG.similarityThreshold).toBeLessThan(lowestCopy);
  });

  it('The_same_logic_written_in_another_language_is_recognised', async () => {
    const embedder = new LocalEmbedder(store);

    const [original, csharp] = await embedder.embed([ORIGINAL, SAME_LOGIC_IN_CSHARP]);

    expect(cosineSimilarity(original, csharp)).toBeGreaterThan(
      DEFAULT_CONFIG.similarityThreshold
    );
  });

  it('The_same_code_embedded_twice_gives_the_same_vector', async () => {
    const embedder = new LocalEmbedder(store);

    const [first, second] = await embedder.embed([ORIGINAL, ORIGINAL]);

    expect(cosineSimilarity(first, second)).toBeGreaterThan(0.999);
  });

  it('Vectors_come_back_normalised_so_similarity_is_a_plain_dot_product', async () => {
    const embedder = new LocalEmbedder(store);

    const [vector] = await embedder.embed([ORIGINAL]);

    let magnitude = 0;
    for (const value of vector) {
      magnitude += value * value;
    }

    expect(Math.sqrt(magnitude)).toBeCloseTo(1, 3);
  });
});

describe('Line alignment on the same fixtures', () => {
  const align = (first: string, second: string) =>
    alignment(contentLines(first), contentLines(second));

  it('A_copy_renamed_throughout_still_lines_up', () => {
    expect(align(ORIGINAL, RENAMED)).toBeGreaterThanOrEqual(MIN_ALIGNMENT);
  });

  it('The_same_logic_in_another_language_still_lines_up', () => {
    expect(align(ORIGINAL, SAME_LOGIC_IN_CSHARP)).toBeGreaterThanOrEqual(MIN_ALIGNMENT);
  });

  it('Code_that_only_shares_a_shape_does_not_line_up', () => {
    expect(align(ORIGINAL, SAME_SHAPE_DIFFERENT_WORK)).toBeLessThan(MIN_ALIGNMENT);
    expect(align(ORIGINAL, UNRELATED)).toBeLessThan(MIN_ALIGNMENT);
  });
});
