import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  GenerationQueue,
  type FileScan,
  type SourceFile,
  type SourceProvider,
} from '../src/embedding/queue.js';
import { EmbeddingCache } from '../src/cache/embedding-cache.js';
import { Chunker } from '../src/chunking/chunker.js';
import { IndexingPace } from '../src/embedding/pacer.js';
import type { Embedder } from '../src/embedding/embedder.js';
import type { Embedding } from '../src/types.js';

/** Stands in for the model, so the queue can be tested without downloading it. */
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

function sourceOf(file: string, lineCount: number, marker: string): SourceFile {
  const lines = [`function ${marker}() {`];
  for (let i = 0; i < lineCount; i++) {
    lines.push(`  ${marker}Step${i}();`);
  }
  lines.push('}');

  return { file, content: lines.join('\n'), mtime: 1000 };
}

/**
 * Presents a list of files the way the real project does: names and stamps
 * first, content only when the queue asks for a particular file.
 */
function providerOver(files: () => SourceFile[]): SourceProvider {
  return {
    list: () => {
      const scans: FileScan[] = [];
      for (const file of files()) {
        scans.push({ file: file.file, mtime: file.mtime, size: file.content.length });
      }
      return scans;
    },
    read: (name) => {
      for (const file of files()) {
        if (file.file === name) return file;
      }
      return null;
    },
  };
}

describe('Embedding generation queue', () => {
  let directory: string;
  let cache: EmbeddingCache;
  let embedder: CountingEmbedder;
  let files: SourceFile[];
  let queue: GenerationQueue;

  beforeEach(() => {
    directory = fs.mkdtempSync(path.join(os.tmpdir(), 'duplication-queue-'));
    cache = new EmbeddingCache(path.join(directory, 'test.db'), 'test-model');
    embedder = new CountingEmbedder();
    files = [sourceOf('src/a.ts', 8, 'alpha'), sourceOf('src/b.ts', 8, 'beta')];
    queue = new GenerationQueue(
      cache,
      new Chunker({ minLines: 3, maxLines: 100 }),
      embedder,
      providerOver(() => files)
    );
  });

  afterEach(() => {
    cache.close();
    fs.rmSync(directory, { recursive: true, force: true });
  });

  it('Files_never_seen_before_are_queued_for_embedding', () => {
    queue.scanForChanges();

    expect(queue.pendingCount()).toBe(2);
  });

  it('The_queue_empties_once_the_work_is_done', async () => {
    queue.scanForChanges();
    await queue.drain();

    expect(queue.pendingCount()).toBe(0);
  });

  it('An_unchanged_file_is_not_embedded_a_second_time', async () => {
    queue.scanForChanges();
    await queue.drain();
    const firstRunCount = embedder.embedded.length;

    queue.scanForChanges();
    await queue.drain();

    expect(embedder.embedded.length).toBe(firstRunCount);
    expect(queue.pendingCount()).toBe(0);
  });

  it('An_edited_file_returns_to_the_queue', async () => {
    queue.scanForChanges();
    await queue.drain();

    files[0] = { ...sourceOf('src/a.ts', 12, 'alpha'), mtime: 2000 };
    queue.scanForChanges();

    expect(queue.pendingCount()).toBe(1);
  });

  it('A_file_that_is_touched_but_not_edited_costs_no_embedding_work', async () => {
    queue.scanForChanges();
    await queue.drain();
    const firstRunCount = embedder.embedded.length;

    files[0] = { ...files[0], mtime: 9999 };
    queue.scanForChanges();
    await queue.drain();

    expect(embedder.embedded.length).toBe(firstRunCount);
  });

  it('Identical_code_in_two_files_is_embedded_only_once', async () => {
    const shared = sourceOf('src/a.ts', 8, 'shared');
    files = [shared, { ...shared, file: 'src/b.ts' }];

    queue.scanForChanges();
    await queue.drain();

    const unique = new Set(embedder.embedded);
    expect(unique.size).toBe(embedder.embedded.length - countDuplicates(embedder.embedded));
    expect(embedder.embedded.length).toBe(unique.size);
  });

  it('A_deleted_file_is_forgotten_so_it_stops_being_reported', async () => {
    queue.scanForChanges();
    await queue.drain();

    files = [files[0]];
    queue.scanForChanges();

    expect(cache.getFileRecord('src/b.ts')).toBeNull();
  });

  it('Asking_to_drain_twice_at_once_does_not_embed_the_same_file_twice', async () => {
    queue.scanForChanges();

    await Promise.all([queue.drain(), queue.drain()]);

    const unique = new Set(embedder.embedded);
    expect(embedder.embedded.length).toBe(unique.size);
  });

  it('Indexing_in_the_background_rests_so_it_does_not_take_the_whole_machine', async () => {
    const { paced, rests } = pacedQueue();

    paced.scanForChanges();
    await paced.drain();

    expect(rests.length).toBeGreaterThan(0);
  });

  it('Indexing_runs_without_resting_while_a_question_is_being_answered', async () => {
    const { paced, pace, rests } = pacedQueue();

    paced.scanForChanges();
    await pace.whileAnswering(() => paced.drain());

    expect(rests).toEqual([]);
  });

  /**
   * A queue that must rest after any work at all, since the fake embedder
   * returns too fast to fill a real slice.
   */
  function pacedQueue() {
    const rests: number[] = [];
    const pace = new IndexingPace({
      workSliceMs: 0,
      sleep: async (ms) => {
        rests.push(ms);
      },
    });

    const paced = new GenerationQueue(
      cache,
      new Chunker({ minLines: 3, maxLines: 100 }),
      embedder,
      providerOver(() => files),
      pace
    );

    return { paced, pace, rests };
  }
});

function countDuplicates(values: string[]): number {
  const seen = new Set<string>();
  let duplicates = 0;
  for (const value of values) {
    if (seen.has(value)) duplicates++;
    seen.add(value);
  }
  return duplicates;
}
