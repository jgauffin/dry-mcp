import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as path from 'path';
import type { Embedding } from '../types.js';

/** What the cache knows about one file's place in the indexing cycle. */
export interface FileRecord {
  file: string;
  mtime: number;
  /** Size in bytes when last seen, which lets a scan skip opening the file. */
  size: number;
  contentHash: string;
  dirty: boolean;
  chunkHashes: string[];
}

/** How many values one SQL statement may carry, well inside SQLite's limit. */
const MAX_PARAMETERS = 500;

/**
 * Remembers embeddings between runs, so a project is only paid for once.
 *
 * Embedding is by far the slowest part of the analysis, and most of a codebase
 * does not change between two questions. Vectors are keyed by the content of
 * the code rather than by its location, which means moving a block to another
 * file, re-indenting it, or adding a comment all reuse the stored vector, and
 * identical blocks in twenty files are embedded once.
 */
export class EmbeddingCache {
  private db: DatabaseSync;
  private modelId: string;

  constructor(databasePath: string, modelId: string) {
    this.modelId = modelId;
    fs.mkdirSync(path.dirname(databasePath), { recursive: true });
    this.db = new DatabaseSync(databasePath);
    this.tune();
    this.createSchema();
  }

  /**
   * Settings that decide how much a write costs.
   *
   * Indexing is thousands of small writes; with the default journal and a
   * flush to disk per transaction it spends most of its time waiting on the
   * disk rather than on the model. The cache can be rebuilt from the code at
   * any time, so trading the strictest durability for that speed is safe.
   */
  private tune(): void {
    this.db.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA synchronous = NORMAL;
      PRAGMA temp_store = MEMORY;
      PRAGMA mmap_size = 268435456;
      PRAGMA cache_size = -16000;
    `);
  }

  private createSchema(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS chunks (
        chunk_hash TEXT PRIMARY KEY,
        vector     BLOB NOT NULL,
        model_id   TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS file_state (
        file         TEXT PRIMARY KEY,
        mtime        REAL NOT NULL,
        size         INTEGER NOT NULL DEFAULT 0,
        content_hash TEXT NOT NULL,
        dirty        INTEGER NOT NULL DEFAULT 1,
        chunk_hashes TEXT NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_file_state_dirty ON file_state(dirty);
    `);

    this.addSizeColumnIfMissing();
  }

  /**
   * Brings a cache written by an earlier version up to date.
   *
   * Dropping it instead would cost the developer a full re-embedding of the
   * project — minutes of CPU — for a column that defaults harmlessly to zero,
   * which merely makes each file be opened once more than strictly needed.
   */
  private addSizeColumnIfMissing(): void {
    const columns = this.db.prepare('PRAGMA table_info(file_state)').all() as { name: string }[];

    for (const column of columns) {
      if (column.name === 'size') return;
    }

    this.db.exec('ALTER TABLE file_state ADD COLUMN size INTEGER NOT NULL DEFAULT 0');
  }

  /**
   * The stored vector for a block, or null when it has never been embedded.
   *
   * A vector produced by a different model is treated as absent: comparing
   * vectors from two models yields meaningless similarities, so switching
   * models has to re-embed rather than silently mix them.
   */
  getVector(chunkHash: string): Embedding | null {
    const row = this.db
      .prepare('SELECT vector FROM chunks WHERE chunk_hash = ? AND model_id = ?')
      .get(chunkHash, this.modelId) as { vector: Uint8Array } | undefined;

    if (!row) return null;

    return toEmbedding(row.vector);
  }

  /**
   * Fetches many vectors at once, skipping those not yet embedded.
   *
   * Asked in batches rather than one query per block: an analysis looks up
   * every block in the project, and a round trip each would dominate the time
   * the whole question takes.
   */
  getVectors(chunkHashes: string[]): Map<string, Embedding> {
    const found = new Map<string, Embedding>();

    for (const batch of inBatches(chunkHashes, MAX_PARAMETERS)) {
      const rows = this.db
        .prepare(
          `SELECT chunk_hash, vector FROM chunks
           WHERE model_id = ? AND chunk_hash IN (${placeholders(batch.length)})`
        )
        .all(this.modelId, ...batch) as { chunk_hash: string; vector: Uint8Array }[];

      for (const row of rows) {
        found.set(row.chunk_hash, toEmbedding(row.vector));
      }
    }

    return found;
  }

  /**
   * Which of these blocks already have a vector.
   *
   * Deciding what still needs embedding only needs the names, and fetching the
   * vectors themselves to answer that would move megabytes for nothing.
   */
  knownVectorHashes(chunkHashes: string[]): Set<string> {
    const known = new Set<string>();

    for (const batch of inBatches(chunkHashes, MAX_PARAMETERS)) {
      const rows = this.db
        .prepare(
          `SELECT chunk_hash FROM chunks
           WHERE model_id = ? AND chunk_hash IN (${placeholders(batch.length)})`
        )
        .all(this.modelId, ...batch) as { chunk_hash: string }[];

      for (const row of rows) {
        known.add(row.chunk_hash);
      }
    }

    return known;
  }

  /** Stores a freshly computed vector so it is never computed again. */
  putVector(chunkHash: string, vector: Embedding): void {
    this.db
      .prepare(
        `INSERT INTO chunks (chunk_hash, vector, model_id, created_at)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(chunk_hash) DO UPDATE SET
           vector = excluded.vector,
           model_id = excluded.model_id,
           created_at = excluded.created_at`
      )
      .run(chunkHash, toBlob(vector), this.modelId, Date.now());
  }

  /** Stores a batch in one transaction, so a crash cannot half-write it. */
  putVectors(entries: { chunkHash: string; vector: Embedding }[]): void {
    if (entries.length === 0) return;

    const statement = this.db.prepare(
      `INSERT INTO chunks (chunk_hash, vector, model_id, created_at)
       VALUES (?, ?, ?, ?)
       ON CONFLICT(chunk_hash) DO UPDATE SET
         vector = excluded.vector,
         model_id = excluded.model_id,
         created_at = excluded.created_at`
    );
    const now = Date.now();

    this.db.exec('BEGIN');
    try {
      for (const entry of entries) {
        statement.run(entry.chunkHash, toBlob(entry.vector), this.modelId, now);
      }
      this.db.exec('COMMIT');
    } catch (error) {
      this.db.exec('ROLLBACK');
      throw error;
    }
  }

  /** What the cache last recorded about a file, if it has seen it before. */
  getFileRecord(file: string): FileRecord | null {
    const row = this.db.prepare('SELECT * FROM file_state WHERE file = ?').get(file) as
      | FileStateRow
      | undefined;

    if (!row) return null;

    return toFileRecord(row);
  }

  /**
   * Records that a file needs embedding, because it is new or has changed.
   *
   * The previously known chunk list is kept so the file can still be reported
   * on from stale data while it waits its turn in the queue.
   */
  markDirty(file: string, mtime: number, size: number, contentHash: string): void {
    this.db
      .prepare(
        `INSERT INTO file_state (file, mtime, size, content_hash, dirty, chunk_hashes)
         VALUES (?, ?, ?, ?, 1, '[]')
         ON CONFLICT(file) DO UPDATE SET
           mtime = excluded.mtime,
           size = excluded.size,
           content_hash = excluded.content_hash,
           dirty = 1`
      )
      .run(file, mtime, size, contentHash);
  }

  /** Records that a file is fully embedded and its blocks are known. */
  markClean(
    file: string,
    mtime: number,
    size: number,
    contentHash: string,
    chunkHashes: string[]
  ): void {
    this.db
      .prepare(
        `INSERT INTO file_state (file, mtime, size, content_hash, dirty, chunk_hashes)
         VALUES (?, ?, ?, ?, 0, ?)
         ON CONFLICT(file) DO UPDATE SET
           mtime = excluded.mtime,
           size = excluded.size,
           content_hash = excluded.content_hash,
           dirty = 0,
           chunk_hashes = excluded.chunk_hashes`
      )
      .run(file, mtime, size, contentHash, JSON.stringify(chunkHashes));
  }

  /** Files still waiting to be embedded. */
  dirtyFiles(): string[] {
    const rows = this.db
      .prepare('SELECT file FROM file_state WHERE dirty = 1 ORDER BY file')
      .all() as { file: string }[];

    const files: string[] = [];
    for (const row of rows) {
      files.push(row.file);
    }
    return files;
  }

  /** How many files are queued, which every answer has to report. */
  pendingCount(): number {
    const row = this.db
      .prepare('SELECT COUNT(*) AS count FROM file_state WHERE dirty = 1')
      .get() as { count: number };
    return row.count;
  }

  /** How many files the cache knows about, without loading any of them. */
  fileCount(): number {
    const row = this.db.prepare('SELECT COUNT(*) AS count FROM file_state').get() as {
      count: number;
    };
    return row.count;
  }

  /** Every file the cache knows about, dirty or not. */
  knownFiles(): FileRecord[] {
    const rows = this.db
      .prepare('SELECT * FROM file_state ORDER BY file')
      .all() as unknown as FileStateRow[];

    const records: FileRecord[] = [];
    for (const row of rows) {
      records.push(toFileRecord(row));
    }
    return records;
  }

  /** Just the names, for callers deciding what still belongs in the index. */
  knownFileNames(): string[] {
    const rows = this.db.prepare('SELECT file FROM file_state').all() as { file: string }[];

    const files: string[] = [];
    for (const row of rows) {
      files.push(row.file);
    }
    return files;
  }

  /** Forgets a file that no longer exists in the project. */
  removeFile(file: string): void {
    this.db.prepare('DELETE FROM file_state WHERE file = ?').run(file);
  }

  /**
   * Forgets every file outside the given set, in one pass.
   *
   * This is how a file leaves the index when it is deleted, or when the
   * project's include and exclude rules stop covering it: it is removed from
   * the queue as well as from what gets reported, so a newly excluded folder
   * costs no further embedding work.
   */
  retainOnly(present: Set<string>): number {
    const doomed: string[] = [];
    for (const file of this.knownFileNames()) {
      if (!present.has(file)) doomed.push(file);
    }

    if (doomed.length === 0) return 0;

    this.db.exec('BEGIN');
    try {
      for (const batch of inBatches(doomed, MAX_PARAMETERS)) {
        this.db
          .prepare(`DELETE FROM file_state WHERE file IN (${placeholders(batch.length)})`)
          .run(...batch);
      }
      this.db.exec('COMMIT');
    } catch (error) {
      this.db.exec('ROLLBACK');
      throw error;
    }

    return doomed.length;
  }

  /** Marks everything for re-embedding, for when the caller wants a clean run. */
  markAllDirty(): void {
    this.db.prepare('UPDATE file_state SET dirty = 1').run();
  }

  chunkCount(): number {
    const row = this.db
      .prepare('SELECT COUNT(*) AS count FROM chunks WHERE model_id = ?')
      .get(this.modelId) as { count: number };
    return row.count;
  }

  /**
   * Discards vectors no live file refers to any more.
   *
   * Without this the cache would grow forever as code is edited, since a vector
   * is keyed by content that may no longer exist anywhere. The live set is
   * built inside the database rather than in memory, because on a large project
   * both lists run to hundreds of thousands of entries.
   */
  pruneOrphanedVectors(): number {
    this.db.exec('BEGIN');
    try {
      this.db.exec('CREATE TEMP TABLE IF NOT EXISTS live_hashes (chunk_hash TEXT PRIMARY KEY)');
      this.db.exec('DELETE FROM live_hashes');

      const insert = this.db.prepare(
        'INSERT OR IGNORE INTO live_hashes (chunk_hash) VALUES (?)'
      );
      const rows = this.db.prepare('SELECT chunk_hashes FROM file_state').all() as {
        chunk_hashes: string;
      }[];

      for (const row of rows) {
        for (const hash of JSON.parse(row.chunk_hashes) as string[]) {
          insert.run(hash);
        }
      }

      const before = this.db.prepare('SELECT COUNT(*) AS count FROM chunks').get() as {
        count: number;
      };
      this.db.exec(
        'DELETE FROM chunks WHERE chunk_hash NOT IN (SELECT chunk_hash FROM live_hashes)'
      );
      const after = this.db.prepare('SELECT COUNT(*) AS count FROM chunks').get() as {
        count: number;
      };

      this.db.exec('DROP TABLE live_hashes');
      this.db.exec('COMMIT');

      return before.count - after.count;
    } catch (error) {
      this.db.exec('ROLLBACK');
      throw error;
    }
  }

  close(): void {
    this.db.close();
  }
}

interface FileStateRow {
  file: string;
  mtime: number;
  size: number;
  content_hash: string;
  dirty: number;
  chunk_hashes: string;
}

function toFileRecord(row: FileStateRow): FileRecord {
  return {
    file: row.file,
    mtime: row.mtime,
    size: row.size ?? 0,
    contentHash: row.content_hash,
    dirty: row.dirty === 1,
    chunkHashes: JSON.parse(row.chunk_hashes) as string[],
  };
}

function placeholders(count: number): string {
  const parts: string[] = [];
  for (let i = 0; i < count; i++) {
    parts.push('?');
  }
  return parts.join(',');
}

function* inBatches<T>(values: T[], size: number): Generator<T[]> {
  for (let start = 0; start < values.length; start += size) {
    yield values.slice(start, start + size);
  }
}

/**
 * Vectors are stored as raw little-endian float32, which is both compact and
 * exactly the layout the similarity maths wants back.
 */
function toBlob(vector: Embedding): Uint8Array {
  return new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength);
}

function toEmbedding(blob: Uint8Array): Embedding {
  // Copy rather than view: the buffer sqlite hands back is not guaranteed to
  // stay valid, and an unaligned offset would make the view throw.
  const copy = new Uint8Array(blob.byteLength);
  copy.set(blob);
  return new Float32Array(copy.buffer);
}
