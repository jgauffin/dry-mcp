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

function duplicatedFunction(marker: string, lines: number): string {
  const body = [`function ${marker}(order) {`];
  for (let i = 0; i < lines; i++) {
    body.push(`  const step${i} = order.lines[${i}].price * 1.25;`);
  }
  body.push('  return order;');
  body.push('}');
  return body.join('\n');
}

/**
 * Embedding a large project takes minutes, and no question waits for it.
 *
 * So an early answer has to say how early it is, and a ranking drawn from a
 * fraction of the code is not offered at all: it would mostly reflect which
 * files the indexer happened to reach first, while reading as an answer.
 */
describe('Answering while the index is still building', () => {
  let projectRoot: string;
  let cacheDirectory: string;
  let service: DuplicationService;

  beforeEach(() => {
    projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'duplication-progress-'));
    cacheDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'duplication-progress-cache-'));
  });

  afterEach(() => {
    service?.dispose();
    fs.rmSync(projectRoot, { recursive: true, force: true });
    fs.rmSync(cacheDirectory, { recursive: true, force: true });
  });

  function startService(): void {
    const cache = new EmbeddingCache(path.join(cacheDirectory, 'test.db'), 'test-model');
    service = new DuplicationService(
      projectRoot,
      { ...DEFAULT_CONFIG, minLines: 3 },
      cache,
      readyModelStore(),
      new StubEmbedder()
    );
    service.refresh();
  }

  /**
   * A project past the small-project threshold is what these cases are about, so
   * writing and embedding a couple of hundred files is a cost they accept.
   */
  const LARGE_PROJECT_TIMEOUT = { timeout: 60_000 };

  function writeFiles(count: number): void {
    const folder = path.join(projectRoot, 'src');
    fs.mkdirSync(folder, { recursive: true });
    for (let i = 0; i < count; i++) {
      fs.writeFileSync(path.join(folder, `file${i}.ts`), duplicatedFunction('shared', 8));
    }
  }

  it(
    'A_large_project_answers_at_once_with_progress_rather_than_a_ranking_of_a_fraction',
    LARGE_PROJECT_TIMEOUT,
    () => {
      writeFiles(250);

      startService();
      const report = service.analyze();

      expect(report.duplications).toHaveLength(0);
      expect(report.progress).toEqual({
        filesInScope: 250,
        filesEmbedded: 0,
        pendingFiles: 250,
        percentComplete: 0,
      });
      expect(report.notice).toContain('0% done');
    }
  );

  it('A_small_project_is_ranked_as_it_stands_because_indexing_it_takes_seconds', async () => {
    const body = duplicatedFunction('shared', 8);
    fs.writeFileSync(path.join(projectRoot, 'a.ts'), body);
    fs.writeFileSync(path.join(projectRoot, 'b.ts'), body);

    startService();
    await service.waitForEmbedding();

    fs.writeFileSync(path.join(projectRoot, 'c.ts'), body);
    service.refresh();

    const report = service.analyze();

    expect(report.duplications.length).toBeGreaterThan(0);
    expect(report.progress?.pendingFiles).toBe(1);
  });

  it('Progress_is_left_out_once_the_whole_project_is_embedded', async () => {
    writeFiles(3);

    startService();
    await service.waitForEmbedding();

    const report = service.analyze();

    expect(report.progress).toBeUndefined();
    expect(report.notice).toBeUndefined();
  });

  it(
    'A_mostly_embedded_large_project_is_ranked_with_the_progress_alongside_it',
    LARGE_PROJECT_TIMEOUT,
    async () => {
      writeFiles(250);

      startService();
      await service.waitForEmbedding();

      // One edited file leaves the index all but complete, which is the case
      // where a ranking is worth having and the queue count is a caveat rather
      // than the whole answer.
      const edited = path.join(projectRoot, 'src', 'file0.ts');
      fs.writeFileSync(edited, duplicatedFunction('edited', 9));
      service.refresh();

      const report = service.analyze();

      expect(report.duplications.length).toBeGreaterThan(0);
      expect(report.progress?.percentComplete).toBe(99);
      expect(report.notice).toContain('1 file(s) are queued');
    }
  );
});
