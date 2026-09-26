import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { GenerationQueue, type SourceFile, type FileScan } from '../src/embedding/queue.js';
import { EmbeddingCache } from '../src/cache/embedding-cache.js';
import { Chunker } from '../src/chunking/chunker.js';
import type { Embedder } from '../src/embedding/embedder.js';
import type { Embedding } from '../src/types.js';

class CountingEmbedder implements Embedder {
  embedded: string[] = [];

  async embed(texts: string[]): Promise<Embedding[]> {
    const vectors: Embedding[] = [];
    for (const text of texts) {
      this.embedded.push(text);
      vectors.push(Float32Array.from([1, 0]));
    }
    return vectors;
  }
}

function bodyOf(lineCount: number, marker: string): string {
  const lines = [`function ${marker}() {`];
  for (let i = 0; i < lineCount; i++) {
    lines.push(`  ${marker}Step${i}();`);
  }
  lines.push('}');
  return lines.join('\n');
}

/**
 * A project whose files can be counted as they are read, which is how the cost
 * of a scan is measured: reading every file to decide whether anything changed
 * is exactly the waste these tests exist to prevent.
 */
class CountingProject {
  private contents = new Map<string, string>();
  private times = new Map<string, number>();
  reads: string[] = [];

  write(file: string, content: string, mtime: number): void {
    this.contents.set(file, content);
    this.times.set(file, mtime);
  }

  remove(file: string): void {
    this.contents.delete(file);
    this.times.delete(file);
  }

  list(): FileScan[] {
    const scans: FileScan[] = [];
    for (const [file, mtime] of this.times) {
      scans.push({ file, mtime, size: this.contents.get(file)!.length });
    }
    return scans;
  }

  read(file: string): SourceFile | null {
    const content = this.contents.get(file);
    if (content === undefined) return null;
    this.reads.push(file);
    return { file, content, mtime: this.times.get(file)! };
  }
}

describe('What indexing costs in time and memory', () => {
  let directory: string;
  let cache: EmbeddingCache;
  let embedder: CountingEmbedder;
  let project: CountingProject;
  let queue: GenerationQueue;

  beforeEach(() => {
    directory = fs.mkdtempSync(path.join(os.tmpdir(), 'duplication-cost-'));
    cache = new EmbeddingCache(path.join(directory, 'test.db'), 'test-model');
    embedder = new CountingEmbedder();
    project = new CountingProject();
    project.write('src/a.ts', bodyOf(8, 'alpha'), 1000);
    project.write('src/b.ts', bodyOf(8, 'beta'), 1000);
    queue = new GenerationQueue(cache, new Chunker({ minLines: 3, maxLines: 100 }), embedder, {
      list: () => project.list(),
      read: (file) => project.read(file),
    });
  });

  afterEach(() => {
    cache.close();
    fs.rmSync(directory, { recursive: true, force: true });
  });

  it('A_scan_of_an_unchanged_project_reads_no_file_contents', async () => {
    queue.scanForChanges();
    await queue.drain();

    project.reads = [];
    queue.scanForChanges();

    expect(project.reads).toEqual([]);
  });

  it('A_file_whose_timestamp_moved_is_read_once_and_not_re_embedded_when_its_content_is_unchanged', async () => {
    queue.scanForChanges();
    await queue.drain();
    const firstRunCount = embedder.embedded.length;

    project.write('src/a.ts', bodyOf(8, 'alpha'), 9999);
    project.reads = [];
    queue.scanForChanges();
    await queue.drain();

    expect(project.reads).toEqual(['src/a.ts']);
    expect(embedder.embedded.length).toBe(firstRunCount);
    expect(queue.pendingCount()).toBe(0);
  });

  it('Draining_touches_only_the_queued_file_rather_than_loading_the_whole_project', async () => {
    project.write('src/c.ts', bodyOf(8, 'gamma'), 1000);
    queue.scanForChanges();
    await queue.drain();

    project.write('src/a.ts', bodyOf(12, 'alpha'), 2000);
    project.reads = [];
    queue.scanForChanges();
    await queue.drain();

    // The one changed file, opened once between the scan and the drain
    // together — the untouched files are never opened at all.
    expect(project.reads).toEqual(['src/a.ts']);
  });

  it('A_file_the_scan_has_already_read_is_not_read_again_to_embed_it', async () => {
    queue.scanForChanges();
    await queue.drain();

    project.write('src/a.ts', bodyOf(12, 'alpha'), 2000);
    queue.scanForChanges();
    project.reads = [];
    await queue.drain();

    expect(project.reads).toEqual([]);
  });

  it('A_requested_rebuild_re_reads_every_file_even_though_none_of_them_changed', async () => {
    queue.scanForChanges();
    await queue.drain();

    // The skip that keeps an ordinary scan cheap must not survive a rebuild:
    // a developer asking for one is saying they no longer trust the index.
    cache.markAllDirty();
    project.reads = [];
    queue.scanForChanges();
    await queue.drain();

    expect(project.reads.sort()).toEqual(['src/a.ts', 'src/b.ts']);
    expect(queue.pendingCount()).toBe(0);
  });

  it('A_file_grown_past_the_size_limit_is_dropped_from_the_index_without_being_read', async () => {
    queue.scanForChanges();
    await queue.drain();

    project.write('src/a.ts', 'x'.repeat(64), 2000);
    project.reads = [];
    queue.scanForChanges({ maxFileBytes: 32 });

    expect(project.reads).toEqual([]);
    expect(cache.getFileRecord('src/a.ts')).toBeNull();
  });
});
