import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DuplicationService } from '../src/analysis/duplication-service.js';
import { EmbeddingCache } from '../src/cache/embedding-cache.js';
import { ModelStore } from '../src/embedding/model-store.js';
import { ToolHandler } from '../src/tools/index.js';
import { DEFAULT_CONFIG } from '../src/config.js';
import type { Embedder } from '../src/embedding/embedder.js';
import type { Embedding } from '../src/types.js';

/**
 * Stands in for the embedding model.
 *
 * The property the real model gives us is that code doing the same thing lands
 * close together even when the names differ. This reproduces that cheaply by
 * describing a block through its structure — keywords, operators and shape —
 * rather than the identifiers a developer happened to choose, which is exactly
 * the distinction a rename should not disturb.
 */
class StubEmbedder implements Embedder {
  async embed(texts: string[]): Promise<Embedding[]> {
    const vectors: Embedding[] = [];
    for (const text of texts) {
      vectors.push(structureVector(text));
    }
    return vectors;
  }
}

const DIMENSIONS = 64;

/** Words that carry meaning wherever they appear, unlike a variable name. */
const KEYWORDS = new Set([
  'function', 'return', 'const', 'let', 'var', 'for', 'of', 'in', 'if', 'else',
  'while', 'do', 'break', 'continue', 'new', 'class', 'this', 'throw', 'try',
  'catch', 'public', 'private', 'void', 'foreach', 'true', 'false', 'null',
]);

function structureVector(text: string): Embedding {
  const vector = new Float32Array(DIMENSIONS);

  for (const token of tokenize(text)) {
    bump(vector, token, 1);
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
 * Reduces code to a sequence where every identifier reads the same, so two
 * blocks differing only in naming produce the same tokens.
 */
function tokenize(text: string): string[] {
  const tokens: string[] = [];
  const raw = text.match(/[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]/g) ?? [];

  for (const piece of raw) {
    if (/^[A-Za-z_]/.test(piece)) {
      tokens.push(KEYWORDS.has(piece.toLowerCase()) ? piece.toLowerCase() : 'name');
    } else {
      tokens.push(piece);
    }
  }

  // Pairs as well as single tokens, so ordering counts for something and two
  // unrelated blocks using the same keywords still come out apart.
  const withPairs = [...tokens];
  for (let i = 0; i + 1 < tokens.length; i++) {
    withPairs.push(`${tokens[i]}~${tokens[i + 1]}`);
  }
  return withPairs;
}

function bump(vector: Float32Array, token: string, weight: number): void {
  let hash = 0;
  for (const char of token) {
    hash = (hash * 31 + char.charCodeAt(0)) % 1000003;
  }
  vector[hash % DIMENSIONS] += weight;
}

const RECONCILE_BLOCK = [
  'function reconcileInvoice(invoice, ledger) {',
  '  const entries = ledger.entriesFor(invoice.id);',
  '  let balance = 0;',
  '  for (const entry of entries) {',
  '    balance += entry.amount;',
  '    if (entry.reversed) {',
  '      balance -= entry.amount * 2;',
  '    }',
  '  }',
  '  invoice.balance = balance;',
  '  invoice.reconciledAt = Date.now();',
  '  return invoice;',
  '}',
].join('\n');

/** The same logic with everything renamed, which only meaning-based matching finds. */
const RENAMED_BLOCK = RECONCILE_BLOCK.replace(/invoice/g, 'bill')
  .replace(/ledger/g, 'book')
  .replace(/balance/g, 'total');

describe('Duplication reported through the tools an agent calls', () => {
  let projectRoot: string;
  let cacheDirectory: string;
  let service: DuplicationService;
  let handler: ToolHandler;

  function start(): void {
    const store = new ModelStore(DEFAULT_CONFIG.model);
    store.status = () => 'ready';
    const cache = new EmbeddingCache(path.join(cacheDirectory, 'e2e.db'), 'stub-model');
    service = new DuplicationService(
      projectRoot,
      DEFAULT_CONFIG,
      cache,
      store,
      new StubEmbedder()
    );
    handler = new ToolHandler(service);
  }

  async function call(name: string, args: Record<string, unknown> = {}): Promise<string> {
    const response = await handler.handleTool(name, args);
    expect(response.isError).not.toBe(true);
    return response.content[0].text;
  }

  beforeEach(() => {
    projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'duplication-e2e-'));
    cacheDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'duplication-e2e-cache-'));
  });

  afterEach(() => {
    handler?.dispose();
    fs.rmSync(projectRoot, { recursive: true, force: true });
    fs.rmSync(cacheDirectory, { recursive: true, force: true });
  });

  it('A_block_copied_into_three_files_is_found_and_every_copy_is_located', async () => {
    fs.writeFileSync(path.join(projectRoot, 'billing.ts'), RECONCILE_BLOCK);
    fs.writeFileSync(path.join(projectRoot, 'invoicing.ts'), RECONCILE_BLOCK);
    fs.mkdirSync(path.join(projectRoot, 'reports'));
    fs.writeFileSync(path.join(projectRoot, 'reports', 'monthly.ts'), RECONCILE_BLOCK);

    start();
    service.refresh();
    await service.waitForEmbedding();

    const report = service.analyze();

    expect(report.duplications).toHaveLength(1);
    expect(report.duplications[0].frequency).toBe(3);
    expect(report.duplications[0].occurrences.map((o) => o.file).sort()).toEqual([
      'billing.ts',
      'invoicing.ts',
      'reports/monthly.ts',
    ]);
  });

  it('A_copy_with_every_name_changed_is_still_recognised_as_the_same_logic', async () => {
    fs.writeFileSync(path.join(projectRoot, 'billing.ts'), RECONCILE_BLOCK);
    fs.writeFileSync(path.join(projectRoot, 'renamed.ts'), RENAMED_BLOCK);

    start();
    service.refresh();
    await service.waitForEmbedding();

    const report = service.analyze();

    expect(report.duplications.length).toBeGreaterThan(0);
    expect(report.duplications[0].matchType).toBe('near-identical');
  });

  it('Asking_while_files_are_queued_reports_how_many_are_waiting', async () => {
    fs.writeFileSync(path.join(projectRoot, 'billing.ts'), RECONCILE_BLOCK);
    fs.writeFileSync(path.join(projectRoot, 'invoicing.ts'), RECONCILE_BLOCK);

    start();
    service.refresh();

    const text = await call('detect_duplication');

    expect(text).toContain('pendingFiles: 2');
    expect(text).toContain('queued for embedding');
  });

  it('Once_the_queue_drains_the_answer_carries_no_caveat', async () => {
    fs.writeFileSync(path.join(projectRoot, 'billing.ts'), RECONCILE_BLOCK);
    fs.writeFileSync(path.join(projectRoot, 'invoicing.ts'), RECONCILE_BLOCK);

    start();
    service.refresh();
    await service.waitForEmbedding();

    const text = await call('detect_duplication');

    expect(text).toContain('pendingFiles: 0');
    expect(text).not.toContain('queued for embedding');
  });

  it('Editing_a_file_makes_the_next_answer_say_it_may_be_out_of_date', async () => {
    const file = path.join(projectRoot, 'billing.ts');
    fs.writeFileSync(file, RECONCILE_BLOCK);
    fs.writeFileSync(path.join(projectRoot, 'invoicing.ts'), RECONCILE_BLOCK);

    start();
    service.refresh();
    await service.waitForEmbedding();

    fs.writeFileSync(file, `${RECONCILE_BLOCK}\n\nfunction addedLater() {\n  return 1;\n}`);
    service.refresh();

    const text = await call('detect_duplication');

    expect(text).toContain('pendingFiles: 1');
  });

  it('A_finding_can_be_traced_to_its_full_source_in_a_follow_up_call', async () => {
    fs.writeFileSync(path.join(projectRoot, 'billing.ts'), RECONCILE_BLOCK);
    fs.writeFileSync(path.join(projectRoot, 'invoicing.ts'), RECONCILE_BLOCK);

    start();
    service.refresh();
    await service.waitForEmbedding();

    const found = service.analyze().duplications[0];
    const explained = await call('explain_duplication', { clusterId: found.id });

    expect(explained).toContain('reconcileInvoice');
  });

  it('Status_says_the_analysis_is_ready_once_nothing_is_queued', async () => {
    fs.writeFileSync(path.join(projectRoot, 'billing.ts'), RECONCILE_BLOCK);

    start();
    service.refresh();
    await service.waitForEmbedding();

    expect(await call('duplication_status')).toContain('ready: true');
  });

  it('Every_finding_says_how_much_it_can_be_trusted', async () => {
    fs.writeFileSync(path.join(projectRoot, 'billing.ts'), RECONCILE_BLOCK);
    fs.writeFileSync(path.join(projectRoot, 'invoicing.ts'), RECONCILE_BLOCK);

    start();
    service.refresh();
    await service.waitForEmbedding();

    const report = service.analyze();

    expect(report.duplications[0].confidence).toBe('certain');
    // The scale travels with the results so the caller need not guess it.
    expect(report.confidenceGuide.certain).toBeTruthy();
    expect(report.summary.byConfidence.certain).toBe(1);
  });

  it('A_caller_wanting_only_safe_findings_can_exclude_the_uncertain_ones', async () => {
    fs.writeFileSync(path.join(projectRoot, 'billing.ts'), RECONCILE_BLOCK);
    fs.writeFileSync(path.join(projectRoot, 'renamed.ts'), RENAMED_BLOCK);

    start();
    service.refresh();
    await service.waitForEmbedding();

    const everything = service.analyze();
    const certainOnly = service.analyze({ minConfidence: 'certain' });

    expect(everything.duplications.length).toBeGreaterThan(0);
    expect(certainOnly.duplications.every((d) => d.confidence === 'certain')).toBe(true);
    expect(certainOnly.duplications.length).toBeLessThanOrEqual(everything.duplications.length);
  });

  it('Demanding_a_near_perfect_match_finds_less_than_the_default_does', async () => {
    fs.writeFileSync(path.join(projectRoot, 'billing.ts'), RECONCILE_BLOCK);
    fs.writeFileSync(path.join(projectRoot, 'renamed.ts'), RENAMED_BLOCK);

    start();
    service.refresh();
    await service.waitForEmbedding();

    const relaxed = service.analyze({ similarityThreshold: 0.3 });
    const strict = service.analyze({ similarityThreshold: 0.999 });

    expect(strict.duplications.length).toBeLessThanOrEqual(relaxed.duplications.length);
  });

  it('Results_are_returned_as_readable_yaml_rather_than_raw_json', async () => {
    fs.writeFileSync(path.join(projectRoot, 'billing.ts'), RECONCILE_BLOCK);

    start();
    service.refresh();

    const text = await call('duplication_status');

    expect(text).toContain('modelStatus: ready');
    expect(text.trimStart().startsWith('{')).toBe(false);
  });
});
