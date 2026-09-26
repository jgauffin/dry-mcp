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
 * When the team changes which files are analysed, the index has to follow.
 *
 * A narrowed exclude list means files that were never looked at now matter, and
 * a widened one means indexed files should stop being reported. Either way the
 * developer's next question must reflect the rules they just wrote, not the
 * ones in force when the server started.
 */
describe('Changing which files are analysed', () => {
  let projectRoot: string;
  let cacheDirectory: string;
  let cache: EmbeddingCache;
  let service: DuplicationService;

  beforeEach(() => {
    projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'duplication-config-'));
    cacheDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'duplication-config-cache-'));
    fs.mkdirSync(path.join(projectRoot, 'src'), { recursive: true });
    fs.mkdirSync(path.join(projectRoot, 'legacy'), { recursive: true });
    fs.writeFileSync(path.join(projectRoot, 'src', 'a.ts'), sourceOf('alpha', 10));
    fs.writeFileSync(path.join(projectRoot, 'legacy', 'b.ts'), sourceOf('beta', 10));

    cache = new EmbeddingCache(path.join(cacheDirectory, 'test.db'), 'test-model');
  });

  afterEach(() => {
    service?.dispose();
    fs.rmSync(projectRoot, { recursive: true, force: true });
    fs.rmSync(cacheDirectory, { recursive: true, force: true });
  });

  function writeConfig(config: { include?: string[]; exclude?: string[] }): void {
    fs.writeFileSync(
      path.join(projectRoot, 'duplication.config.json'),
      JSON.stringify({ minLines: 3, ...config })
    );
  }

  function startService(): void {
    service = new DuplicationService(
      projectRoot,
      { ...DEFAULT_CONFIG, minLines: 3, exclude: ['legacy/**'] },
      cache,
      readyModelStore(),
      new StubEmbedder()
    );
  }

  it('Files_newly_brought_into_scope_by_a_narrowed_exclude_list_are_queued', async () => {
    writeConfig({ exclude: ['legacy/**'] });
    startService();
    service.refresh();
    await service.waitForEmbedding();
    expect(service.status().pendingFiles).toBe(0);

    writeConfig({ exclude: [] });
    service.refresh();

    expect(service.status().pendingFiles).toBe(1);
  });

  it('Files_pushed_out_of_scope_by_a_widened_exclude_list_leave_the_queue', async () => {
    writeConfig({ exclude: [] });
    startService();
    service.refresh();
    await service.waitForEmbedding();
    expect(service.status().filesIndexed).toBe(2);

    writeConfig({ exclude: ['legacy/**'] });
    service.refresh();

    expect(service.status().filesIndexed).toBe(1);
    expect(service.status().pendingFiles).toBe(0);
  });

  it('A_file_still_waiting_in_the_queue_is_dropped_when_an_exclude_rule_reaches_it', () => {
    writeConfig({ exclude: [] });
    startService();
    service.refresh();
    expect(service.status().pendingFiles).toBe(2);

    // The developer excludes the folder before the queue ever gets to it.
    writeConfig({ exclude: ['legacy/**'] });
    service.refresh();

    expect(service.status().pendingFiles).toBe(1);
    expect(cache.getFileRecord('legacy/b.ts')).toBeNull();
  });

  it('A_file_excluded_while_queued_is_never_embedded', async () => {
    writeConfig({ exclude: [] });
    startService();
    service.refresh();

    writeConfig({ exclude: ['legacy/**'] });
    service.refresh();
    await service.waitForEmbedding();

    expect(cache.getFileRecord('legacy/b.ts')).toBeNull();
    expect(service.status().filesIndexed).toBe(1);
  });

  it('A_narrowed_include_list_drops_the_files_it_no_longer_names', async () => {
    writeConfig({ include: ['**'] });
    startService();
    service.refresh();
    await service.waitForEmbedding();
    expect(service.status().filesIndexed).toBe(2);

    writeConfig({ include: ['src/**'] });
    service.refresh();

    expect(service.status().filesIndexed).toBe(1);
  });

  it('An_unchanged_config_file_leaves_the_queue_alone', async () => {
    writeConfig({ exclude: ['legacy/**'] });
    startService();
    service.refresh();
    await service.waitForEmbedding();

    service.refresh();

    expect(service.status().pendingFiles).toBe(0);
  });
});
