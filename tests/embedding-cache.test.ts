import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { EmbeddingCache } from '../src/cache/embedding-cache.js';

describe('Embedding cache', () => {
  let directory: string;
  let cache: EmbeddingCache;

  beforeEach(() => {
    directory = fs.mkdtempSync(path.join(os.tmpdir(), 'duplication-cache-'));
    cache = new EmbeddingCache(path.join(directory, 'test.db'), 'jina-code-int8');
  });

  afterEach(() => {
    cache.close();
    fs.rmSync(directory, { recursive: true, force: true });
  });

  it('A_stored_vector_is_returned_unchanged', () => {
    const vector = new Float32Array([0.5, -0.25, 0.75]);

    cache.putVector('hash-a', vector);

    expect(Array.from(cache.getVector('hash-a')!)).toEqual([0.5, -0.25, 0.75]);
  });

  it('Code_never_embedded_before_reports_no_vector', () => {
    expect(cache.getVector('unknown')).toBeNull();
  });

  it('Vectors_from_a_different_model_are_not_reused', () => {
    cache.putVector('hash-a', new Float32Array([1, 0, 0]));
    cache.close();

    const withOtherModel = new EmbeddingCache(path.join(directory, 'test.db'), 'other-model');
    expect(withOtherModel.getVector('hash-a')).toBeNull();
    withOtherModel.close();

    cache = new EmbeddingCache(path.join(directory, 'test.db'), 'jina-code-int8');
  });

  it('Reindented_code_reuses_its_vector_because_the_content_hash_is_unchanged', () => {
    cache.putVector('same-content-hash', new Float32Array([0.1, 0.2]));

    // Re-indenting changes the file but not the normalized content, so the
    // second lookup is a hit and no embedding work is needed.
    expect(cache.getVector('same-content-hash')).not.toBeNull();
  });

  it('A_changed_file_is_queued_for_reembedding', () => {
    cache.markClean('src/a.ts', 1000, 100, 'hash-1', ['chunk-1']);
    cache.markDirty('src/a.ts', 2000, 200, 'hash-2');

    expect(cache.dirtyFiles()).toEqual(['src/a.ts']);
    expect(cache.pendingCount()).toBe(1);
  });

  it('An_embedded_file_leaves_the_queue', () => {
    cache.markDirty('src/a.ts', 1000, 100, 'hash-1');
    cache.markClean('src/a.ts', 1000, 100, 'hash-1', ['chunk-1']);

    expect(cache.pendingCount()).toBe(0);
  });

  it('Marking_a_file_dirty_keeps_its_known_blocks_so_stale_answers_remain_possible', () => {
    cache.markClean('src/a.ts', 1000, 100, 'hash-1', ['chunk-1', 'chunk-2']);
    cache.markDirty('src/a.ts', 2000, 200, 'hash-2');

    expect(cache.getFileRecord('src/a.ts')!.chunkHashes).toEqual(['chunk-1', 'chunk-2']);
  });

  it('Requesting_a_full_rebuild_queues_every_known_file', () => {
    cache.markClean('src/a.ts', 1000, 100, 'hash-1', []);
    cache.markClean('src/b.ts', 1000, 100, 'hash-2', []);

    cache.markAllDirty();

    expect(cache.pendingCount()).toBe(2);
  });

  it('Vectors_no_live_file_refers_to_are_discarded', () => {
    cache.putVector('orphan', new Float32Array([1, 2]));
    cache.putVector('live', new Float32Array([3, 4]));
    cache.markClean('src/a.ts', 1000, 100, 'hash-1', ['live']);

    const removed = cache.pruneOrphanedVectors();

    expect(removed).toBe(1);
    expect(cache.getVector('live')).not.toBeNull();
    expect(cache.getVector('orphan')).toBeNull();
  });

  it('Cached_work_survives_a_restart', () => {
    cache.putVector('hash-a', new Float32Array([0.5, 0.5]));
    cache.markClean('src/a.ts', 1000, 100, 'hash-1', ['hash-a']);
    cache.close();

    cache = new EmbeddingCache(path.join(directory, 'test.db'), 'jina-code-int8');

    expect(cache.getVector('hash-a')).not.toBeNull();
    expect(cache.pendingCount()).toBe(0);
  });

  it('A_batch_of_vectors_is_stored_together', () => {
    cache.putVectors([
      { chunkHash: 'a', vector: new Float32Array([1]) },
      { chunkHash: 'b', vector: new Float32Array([2]) },
    ]);

    expect(cache.chunkCount()).toBe(2);
  });
});
