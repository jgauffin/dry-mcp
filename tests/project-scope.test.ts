import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DuplicationService } from '../src/analysis/duplication-service.js';
import { EmbeddingCache } from '../src/cache/embedding-cache.js';
import { ModelStore } from '../src/embedding/model-store.js';
import { DEFAULT_CONFIG, type DuplicationConfig } from '../src/config.js';
import type { Embedder } from '../src/embedding/embedder.js';
import type { Embedding } from '../src/types.js';

class StubEmbedder implements Embedder {
  async embed(texts: string[]): Promise<Embedding[]> {
    const vectors: Embedding[] = [];
    for (const text of texts) {
      vectors.push(Float32Array.from([text.length % 7, 1]));
    }
    return vectors;
  }
}

function readyModelStore(): ModelStore {
  const store = new ModelStore(DEFAULT_CONFIG.model);
  store.status = () => 'ready';
  return store;
}

function sourceOf(marker: string, lines: number): string {
  const body = [`function ${marker}(order) {`];
  for (let i = 0; i < lines; i++) {
    body.push(`  const step${i} = order.lines[${i}].price * 1.25;`);
  }
  body.push('  return order;');
  body.push('}');
  return body.join('\n');
}

/**
 * What gets analysed, and whether a caller can tell.
 *
 * The first run in a repository has no configuration, so everything under the
 * root is in scope — including the submodules, which are other projects. The
 * answers have to say so, and say what to write to change it, or the caller
 * spends the indexing time before discovering the scope was wrong.
 */
describe('Deciding what is in scope', () => {
  let projectRoot: string;
  let cacheDirectory: string;
  let service: DuplicationService;

  beforeEach(() => {
    projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'duplication-scope-'));
    cacheDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'duplication-scope-cache-'));
  });

  afterEach(() => {
    service?.dispose();
    fs.rmSync(projectRoot, { recursive: true, force: true });
    fs.rmSync(cacheDirectory, { recursive: true, force: true });
  });

  function startService(overrides: Partial<DuplicationConfig> = {}): void {
    const cache = new EmbeddingCache(path.join(cacheDirectory, 'test.db'), 'test-model');
    service = new DuplicationService(
      projectRoot,
      { ...DEFAULT_CONFIG, minLines: 3, ...overrides },
      cache,
      readyModelStore(),
      new StubEmbedder()
    );
    service.refresh();
  }

  /** A checked-out submodule: a directory whose .git points at the parent's storage. */
  function writeSubmodule(directory: string, marker: string): void {
    const full = path.join(projectRoot, directory);
    fs.mkdirSync(full, { recursive: true });
    fs.writeFileSync(path.join(full, '.git'), `gitdir: ../.git/modules/${directory}\n`);
    fs.writeFileSync(path.join(full, 'code.ts'), sourceOf(marker, 10));
  }

  /**
   * Long enough for the cases that need a project past the small-project
   * threshold: writing a couple of hundred files is the cost of testing what
   * only happens above it.
   */
  const LARGE_PROJECT_TIMEOUT = { timeout: 30_000 };

  /** Enough files that the project counts as one worth describing the scope of. */
  function writeManyFiles(folder: string, count: number): void {
    const full = path.join(projectRoot, folder);
    fs.mkdirSync(full, { recursive: true });
    for (let i = 0; i < count; i++) {
      fs.writeFileSync(path.join(full, `file${i}.ts`), sourceOf(`marker${i}`, 8));
    }
  }

  it('A_git_submodule_is_another_project_and_is_left_out_of_the_analysis', () => {
    fs.writeFileSync(path.join(projectRoot, 'own.ts'), sourceOf('own', 10));
    writeSubmodule('libs/vendored', 'theirs');

    startService();

    expect(service.status().filesIndexed).toBe(1);
    expect(service.scopeSummary().skippedNestedRepositories).toEqual(['libs/vendored']);
  });

  it('Submodule_code_is_analysed_when_the_team_asks_for_it_explicitly', () => {
    fs.writeFileSync(path.join(projectRoot, 'own.ts'), sourceOf('own', 10));
    writeSubmodule('libs/vendored', 'theirs');

    startService({ includeNestedRepositories: true });

    expect(service.status().filesIndexed).toBe(2);
    expect(service.scopeSummary().skippedNestedRepositories).toBeUndefined();
  });

  it('A_skipped_submodule_is_named_in_the_reply_with_how_to_include_it', () => {
    fs.writeFileSync(path.join(projectRoot, 'own.ts'), sourceOf('own', 10));
    writeSubmodule('libs/vendored', 'theirs');

    startService();
    const report = service.analyze();

    expect(report.notice).toContain('libs/vendored');
    expect(report.notice).toContain('includeNestedRepositories');
    expect(report.scope?.skippedNestedRepositories).toEqual(['libs/vendored']);
  });

  it(
    'An_unconfigured_project_is_told_which_file_to_write_to_narrow_the_scope',
    LARGE_PROJECT_TIMEOUT,
    () => {
      writeManyFiles('third-party', 210);

      startService();
      const report = service.analyze();

      expect(report.notice).toContain('duplication.config.json');
      expect(report.notice).toContain('exclude');
      expect(report.scope?.configured).toBe(false);
      expect(report.scope?.largestFolders[0]).toEqual({ folder: 'third-party', files: 210 });
    }
  );

  it(
    'A_project_that_has_written_its_scope_down_is_not_told_how_to_narrow_it',
    LARGE_PROJECT_TIMEOUT,
    () => {
      writeManyFiles('src', 210);
      fs.writeFileSync(
        path.join(projectRoot, 'duplication.config.json'),
        JSON.stringify({ minLines: 3, include: ['src/**'] })
      );

      startService();
      const report = service.analyze();

      expect(report.notice ?? '').not.toContain('duplication.config.json');
      expect(report.scope).toBeUndefined();
    }
  );

  it('A_small_unconfigured_project_is_left_alone_rather_than_lectured_about_scope', () => {
    fs.writeFileSync(path.join(projectRoot, 'a.ts'), sourceOf('alpha', 10));

    startService();
    const report = service.analyze();

    expect(report.scope).toBeUndefined();
    expect(report.notice ?? '').not.toContain('duplication.config.json');
  });
});
