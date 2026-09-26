import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DuplicationService } from '../src/analysis/duplication-service.js';
import { EmbeddingCache } from '../src/cache/embedding-cache.js';
import { ModelStore } from '../src/embedding/model-store.js';
import { DEFAULT_CONFIG } from '../src/config.js';
import type { Embedder } from '../src/embedding/embedder.js';
import type { Embedding } from '../src/types.js';

/**
 * Stands in for the model so the analysis can be tested without downloading it.
 *
 * Similarity has to mean something for these tests to be worth anything, so
 * each block becomes a vector over the words it contains. Two blocks sharing
 * most of their vocabulary come out close, and unrelated blocks come out far
 * apart — the property the real model provides, cheaply enough to run in a
 * unit test.
 */
class StubEmbedder implements Embedder {
  async embed(texts: string[]): Promise<Embedding[]> {
    const vectors: Embedding[] = [];
    for (const text of texts) {
      vectors.push(vocabularyVector(text));
    }
    return vectors;
  }
}

/** Number of buckets the vocabulary is spread across. */
const DIMENSIONS = 64;

/** A unit vector describing which words a block uses. */
function vocabularyVector(text: string): Embedding {
  const vector = new Float32Array(DIMENSIONS);
  const words = text.toLowerCase().match(/[a-z_][a-z0-9_]*/g) ?? [];

  for (const word of words) {
    let hash = 0;
    for (const char of word) {
      hash = (hash * 31 + char.charCodeAt(0)) % 1000003;
    }
    vector[hash % DIMENSIONS] += 1;
  }

  let magnitude = 0;
  for (const value of vector) {
    magnitude += value * value;
  }
  magnitude = Math.sqrt(magnitude);
  if (magnitude === 0) return vector;

  for (let i = 0; i < vector.length; i++) {
    vector[i] /= magnitude;
  }
  return vector;
}

/**
 * Places each block at a chosen angle, so the similarity between any two of them
 * is exact and a threshold can be aimed deliberately between them.
 */
class FixedAngleEmbedder implements Embedder {
  async embed(texts: string[]): Promise<Embedding[]> {
    const vectors: Embedding[] = [];
    for (const text of texts) {
      const radians = (degreesFor(text) * Math.PI) / 180;
      vectors.push(new Float32Array([Math.cos(radians), Math.sin(radians)]));
    }
    return vectors;
  }
}

/**
 * Where each marked block sits. Alpha and beta are 20 degrees apart (0.94) while
 * gamma is 60 from alpha (0.5), so a threshold of 0.8 groups the pair and the
 * default 0.45 takes in all three.
 */
const BLOCK_ANGLES: Record<string, number> = { alpha: 0, beta: 20, gamma: 60 };

function degreesFor(text: string): number {
  for (const [marker, degrees] of Object.entries(BLOCK_ANGLES)) {
    if (text.includes(marker)) return degrees;
  }
  return 90;
}

/** A model store that claims readiness, for tests that never load the model. */
function readyModelStore(): ModelStore {
  const store = new ModelStore(DEFAULT_CONFIG.model);
  store.status = () => 'ready';
  return store;
}

function duplicatedFunction(marker: string, lines: number): string {
  const body = [`function ${marker}(order) {`];
  for (let i = 0; i < lines; i++) {
    body.push(`  const step${i} = order.lines[${i}].price * 1.25;`);
  }
  body.push('  return order;');
  body.push('}');
  return body.join('\n');
}

describe('Duplication analysis', () => {
  let projectRoot: string;
  let cacheDirectory: string;
  let service: DuplicationService;

  function buildService(
    store: ModelStore = readyModelStore(),
    embedder: Embedder = new StubEmbedder()
  ): DuplicationService {
    const cache = new EmbeddingCache(path.join(cacheDirectory, 'test.db'), 'test-model');
    return new DuplicationService(projectRoot, DEFAULT_CONFIG, cache, store, embedder);
  }

  beforeEach(() => {
    projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'duplication-project-'));
    cacheDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'duplication-cache-'));
  });

  afterEach(() => {
    service?.dispose();
    fs.rmSync(projectRoot, { recursive: true, force: true });
    fs.rmSync(cacheDirectory, { recursive: true, force: true });
  });

  it('The_same_block_copied_into_two_files_is_reported_as_duplication', async () => {
    const body = duplicatedFunction('applyDiscount', 8);
    fs.writeFileSync(path.join(projectRoot, 'a.ts'), body);
    fs.writeFileSync(path.join(projectRoot, 'b.ts'), body);

    service = buildService();
    service.refresh();
    await service.waitForEmbedding();

    const report = service.analyze();

    expect(report.duplications.length).toBeGreaterThan(0);
    expect(report.duplications[0].frequency).toBeGreaterThanOrEqual(2);
  });

  it('A_project_without_repeated_code_reports_nothing_to_fix', async () => {
    const pricing = [
      'function applyDiscount(order) {',
      '  const rate = lookupCustomerRate(order.customerId);',
      '  for (const line of order.lines) {',
      '    line.price = line.price * (1 - rate);',
      '  }',
      '  order.discountApplied = true;',
      '  return order;',
      '}',
    ].join('\n');

    const parsing = [
      'function parseConfiguration(raw) {',
      '  const document = JSON.parse(raw);',
      '  if (!document.version) {',
      '    throw new ConfigurationError("missing version");',
      '  }',
      '  validateSchema(document);',
      '  return document;',
      '}',
    ].join('\n');

    fs.writeFileSync(path.join(projectRoot, 'a.ts'), pricing);
    fs.writeFileSync(path.join(projectRoot, 'b.ts'), parsing);

    service = buildService();
    service.refresh();
    await service.waitForEmbedding();

    expect(service.analyze().duplications).toHaveLength(0);
  });

  it('Files_waiting_to_be_embedded_are_counted_in_the_reply', () => {
    fs.writeFileSync(path.join(projectRoot, 'a.ts'), duplicatedFunction('pending', 8));

    service = buildService();
    service.refresh();

    const report = service.analyze();

    expect(report.status.pendingFiles).toBe(1);
    expect(report.notice).toContain('1 file(s) are queued');
  });

  it('A_fully_indexed_project_reports_no_queue_and_no_caveat', async () => {
    fs.writeFileSync(path.join(projectRoot, 'a.ts'), duplicatedFunction('done', 8));

    service = buildService();
    service.refresh();
    await service.waitForEmbedding();

    const report = service.analyze();

    expect(report.status.pendingFiles).toBe(0);
    expect(report.notice).toBeUndefined();
  });

  it('Editing_a_file_puts_it_back_in_the_queue_and_the_reply_says_so', async () => {
    const file = path.join(projectRoot, 'a.ts');
    fs.writeFileSync(file, duplicatedFunction('original', 8));

    service = buildService();
    service.refresh();
    await service.waitForEmbedding();

    fs.writeFileSync(file, duplicatedFunction('edited', 14));
    service.refresh();

    const report = service.analyze();

    expect(report.status.pendingFiles).toBe(1);
    expect(report.notice).toContain('queued for embedding');
  });

  it('Without_the_model_the_answer_is_empty_and_explains_how_to_install_it', () => {
    fs.writeFileSync(path.join(projectRoot, 'a.ts'), duplicatedFunction('anything', 8));

    const missingModel = new ModelStore(DEFAULT_CONFIG.model);
    missingModel.status = () => 'not-installed';

    service = buildService(missingModel);
    service.refresh();

    const report = service.analyze();

    expect(report.duplications).toHaveLength(0);
    expect(report.status.modelStatus).toBe('not-installed');
    expect(report.notice).toContain('download-model');
  });

  it('An_interrupted_download_is_reported_as_incomplete_rather_than_missing', () => {
    const partial = new ModelStore(DEFAULT_CONFIG.model);
    partial.status = () => 'incomplete';

    service = buildService(partial);

    expect(service.analyze().notice).toContain('incomplete');
  });

  it('Large_duplication_is_ranked_above_a_small_one_repeated_more_often', async () => {
    const large = duplicatedFunction('largeBlock', 30);
    fs.writeFileSync(path.join(projectRoot, 'large-a.ts'), large);
    fs.writeFileSync(path.join(projectRoot, 'large-b.ts'), large);

    const small = duplicatedFunction('smallBlock', 5);
    for (let i = 0; i < 6; i++) {
      fs.writeFileSync(path.join(projectRoot, `small-${i}.ts`), small);
    }

    service = buildService();
    service.refresh();
    await service.waitForEmbedding();

    const report = service.analyze();

    expect(report.duplications[0].medianLines).toBeGreaterThan(20);
  });

  it('Ordering_by_frequency_puts_the_most_widespread_copy_first', async () => {
    const large = duplicatedFunction('largeBlock', 30);
    fs.writeFileSync(path.join(projectRoot, 'large-a.ts'), large);
    fs.writeFileSync(path.join(projectRoot, 'large-b.ts'), large);

    const small = duplicatedFunction('smallBlock', 6);
    for (let i = 0; i < 6; i++) {
      fs.writeFileSync(path.join(projectRoot, `small-${i}.ts`), small);
    }

    service = buildService();
    service.refresh();
    await service.waitForEmbedding();

    const report = service.analyze({ orderBy: 'frequency' });

    expect(report.duplications[0].frequency).toBe(6);
  });

  it('The_full_source_of_a_finding_can_be_fetched_to_act_on_it', async () => {
    const body = duplicatedFunction('fetchable', 8);
    fs.writeFileSync(path.join(projectRoot, 'a.ts'), body);
    fs.writeFileSync(path.join(projectRoot, 'b.ts'), body);

    service = buildService();
    service.refresh();
    await service.waitForEmbedding();

    const found = service.analyze().duplications[0];
    const explained = service.explain(found.id);

    expect(explained).not.toBeNull();
    expect(explained!.sources.length).toBe(found.occurrences.length);
    expect(explained!.sources[0].text).toContain('fetchable');
  });

  it('A_finding_reported_under_a_stricter_threshold_can_still_be_explained', async () => {
    fs.writeFileSync(path.join(projectRoot, 'a.ts'), duplicatedFunction('alpha', 8));
    fs.writeFileSync(path.join(projectRoot, 'b.ts'), duplicatedFunction('beta', 8));
    fs.writeFileSync(path.join(projectRoot, 'c.ts'), duplicatedFunction('gamma', 8));

    service = buildService(readyModelStore(), new FixedAngleEmbedder());
    service.refresh();
    await service.waitForEmbedding();

    const strict = service.analyze({ similarityThreshold: 0.8 });

    expect(strict.duplications.length).toBeGreaterThan(0);
    for (const finding of strict.duplications) {
      expect(service.explain(finding.id), `finding ${finding.id}`).not.toBeNull();
    }
  });

  it('A_finding_whose_copies_are_gone_is_reported_as_gone_rather_than_from_memory', async () => {
    const body = duplicatedFunction('removed', 8);
    fs.writeFileSync(path.join(projectRoot, 'a.ts'), body);
    fs.writeFileSync(path.join(projectRoot, 'b.ts'), body);

    service = buildService();
    service.refresh();
    await service.waitForEmbedding();
    const found = service.analyze().duplications[0];

    fs.rmSync(path.join(projectRoot, 'b.ts'));

    expect(service.explain(found.id)).toBeNull();
  });

  it('Asking_about_a_finding_that_no_longer_exists_explains_itself', () => {
    service = buildService();
    service.refresh();

    const explained = service.explain('nonexistent');

    expect(explained).toBeNull();
  });

  it('Naming_the_paths_to_analyse_leaves_everything_else_out', async () => {
    const body = duplicatedFunction('shared', 8);
    fs.mkdirSync(path.join(projectRoot, 'src'), { recursive: true });
    fs.mkdirSync(path.join(projectRoot, 'legacy'), { recursive: true });
    fs.writeFileSync(path.join(projectRoot, 'src', 'a.ts'), body);
    fs.writeFileSync(path.join(projectRoot, 'src', 'b.ts'), body);
    // Duplicated just as badly, but outside the paths asked for.
    fs.writeFileSync(path.join(projectRoot, 'legacy', 'x.ts'), body);
    fs.writeFileSync(path.join(projectRoot, 'legacy', 'y.ts'), body);

    const store = readyModelStore();
    const cache = new EmbeddingCache(path.join(cacheDirectory, 'whitelist.db'), 'test-model');
    service = new DuplicationService(
      projectRoot,
      { ...DEFAULT_CONFIG, include: ['src/**'] },
      cache,
      store,
      new StubEmbedder()
    );
    service.refresh();
    await service.waitForEmbedding();

    const files: string[] = [];
    for (const cluster of service.analyze().duplications) {
      for (const occurrence of cluster.occurrences) {
        files.push(occurrence.file);
      }
    }

    expect(files.length).toBeGreaterThan(0);
    expect(files.every((file) => file.startsWith('src/'))).toBe(true);
  });

  it('Excluded_paths_are_left_out_of_the_analysis_entirely', async () => {
    const body = duplicatedFunction('generated', 8);
    fs.mkdirSync(path.join(projectRoot, 'migrations'), { recursive: true });
    fs.writeFileSync(path.join(projectRoot, 'migrations', 'a.ts'), body);
    fs.writeFileSync(path.join(projectRoot, 'migrations', 'b.ts'), body);

    service = buildService();
    service.refresh();
    await service.waitForEmbedding();

    expect(service.analyze().duplications).toHaveLength(0);
  });

  it('Duplication_across_different_languages_is_still_found', async () => {
    const body = [
      'public void Apply(Order order) {',
      '    var total = 0;',
      '    foreach (var line in order.Lines) {',
      '        total += line.Price;',
      '    }',
      '    order.Total = total;',
      '}',
    ].join('\n');

    fs.writeFileSync(path.join(projectRoot, 'a.cs'), body);
    fs.writeFileSync(path.join(projectRoot, 'b.cs'), body);

    service = buildService();
    service.refresh();
    await service.waitForEmbedding();

    expect(service.analyze().duplications.length).toBeGreaterThan(0);
  });

  it('Requesting_a_rebuild_queues_every_file_again', async () => {
    fs.writeFileSync(path.join(projectRoot, 'a.ts'), duplicatedFunction('rebuilt', 8));

    service = buildService();
    service.refresh();
    await service.waitForEmbedding();

    service.reindex();

    expect(service.status().pendingFiles).toBe(1);
  });
});
