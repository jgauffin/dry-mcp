import { createHash } from 'crypto';
import type { Chunk, Embedding } from '../types.js';
import type { EmbeddingCache } from '../cache/embedding-cache.js';
import type { Chunker } from '../chunking/chunker.js';
import type { Embedder } from './embedder.js';

/** What the index needs to know about a file to decide whether it has changed. */
export interface SourceFile {
  file: string;
  content: string;
  mtime: number;
}

/**
 * What can be learned about a file without opening it.
 *
 * Listing a project this way costs one directory walk, while reading every
 * file costs the whole codebase in memory. Since a scan runs before every
 * question, the cheap form is what the scan is built on.
 */
export interface FileScan {
  file: string;
  mtime: number;
  size: number;
}

/** Supplies the current state of the project's source, a file at a time. */
export interface SourceProvider {
  /** Every file in scope, without reading any of them. */
  list(): FileScan[];
  /** One file's content, or null when it has gone or cannot be read. */
  read(file: string): SourceFile | null;
}

/** Limits a scan applies when deciding what belongs in the index. */
export interface ScanLimits {
  /** Files larger than this are not worth embedding, so they never enter the index. */
  maxFileBytes?: number;
}

/**
 * Keeps embeddings up to date with the code, in the background.
 *
 * Embedding a whole project takes minutes, which no developer will wait for
 * mid-question. So changed files are queued rather than embedded on demand, and
 * questions are answered from whatever is ready. The cost of that choice is
 * that an answer can be out of date, which is why the number of files still
 * queued travels with every reply — a developer who knows twelve files are
 * pending can judge the result, while one who is not told cannot.
 *
 * Only one file's text is ever held at a time. Loading the project at once
 * would be simpler, but on a large codebase it is hundreds of megabytes that
 * are read again on every question and thrown away unused.
 */
export class GenerationQueue {
  private cache: EmbeddingCache;
  private chunkerFor: () => Chunker;
  private embedder: Embedder;
  private source: SourceProvider;
  private running: Promise<void> | null = null;
  private stopRequested = false;
  /** Files the scan has already read, waiting to be embedded without a second read. */
  private held = new Map<string, SourceFile>();

  /**
   * The chunker is asked for rather than held, because the settings that shape
   * blocks can change while the server runs and the queue must use the rules
   * the developer has in force now.
   */
  constructor(
    cache: EmbeddingCache,
    chunker: Chunker | (() => Chunker),
    embedder: Embedder,
    source: SourceProvider
  ) {
    this.cache = cache;
    this.chunkerFor = typeof chunker === 'function' ? chunker : () => chunker;
    this.embedder = embedder;
    this.source = source;
  }

  /**
   * Compares the project against what was last embedded and queues the
   * difference. Files that vanished, or that the project's rules no longer
   * cover, are forgotten so they stop being reported and stop being embedded.
   *
   * A file is only opened when its timestamp or size says it may have changed.
   * That check is what keeps a scan cheap enough to run before every question:
   * on an untouched project it reads nothing at all.
   */
  scanForChanges(limits: ScanLimits = {}): void {
    const scans = this.source.list();
    const present = new Set<string>();

    for (const scan of scans) {
      if (limits.maxFileBytes !== undefined && scan.size > limits.maxFileBytes) continue;

      present.add(scan.file);
      const record = this.cache.getFileRecord(scan.file);

      // An untouched file cannot have changed, so it is not worth opening.
      // Size guards the case of an edit landing inside the timestamp's
      // resolution, which on some filesystems is a whole second.
      if (record && !record.dirty && record.mtime === scan.mtime && record.size === scan.size) {
        continue;
      }

      const file = this.source.read(scan.file);
      if (!file) {
        present.delete(scan.file);
        continue;
      }

      // Content decides, not mtime: a file merely touched should not cost an
      // embedding run, and a file reverted to its old content is already
      // covered by what was embedded before.
      const contentHash = hashContent(file.content);
      if (record && !record.dirty && record.contentHash === contentHash) {
        // Nothing to embed, but remember what the file looks like now so the
        // next scan can skip it without opening it again.
        this.cache.markClean(file.file, file.mtime, scan.size, contentHash, record.chunkHashes);
        continue;
      }

      this.cache.markDirty(file.file, file.mtime, scan.size, contentHash);
      this.holdForDrain(file);
    }

    this.cache.retainOnly(present);
    this.forgetHeldFilesOutside(present);
  }

  /**
   * Keeps a file the scan has just read, so the drain does not open it again.
   *
   * The scan reads a changed file to hash it and the drain needs that same
   * text moments later — on a first index that is every file in the project
   * read twice. Only a small number are kept, because the whole point of the
   * queue is that the project does not fit comfortably in memory.
   */
  private holdForDrain(file: SourceFile): void {
    if (this.held.size >= MAX_HELD_FILES) return;
    if (Buffer.byteLength(file.content, 'utf-8') > MAX_HELD_FILE_BYTES) return;

    this.held.set(file.file, file);
  }

  private forgetHeldFilesOutside(present: Set<string>): void {
    for (const file of this.held.keys()) {
      if (!present.has(file)) this.held.delete(file);
    }
  }

  /**
   * The file's text, from the scan that just read it when possible.
   *
   * Either way the text is released as it is handed over, so nothing stays in
   * memory once the file has been embedded.
   */
  private takeContent(path: string): SourceFile | null {
    const held = this.held.get(path);
    if (held) {
      this.held.delete(path);
      return held;
    }

    return this.source.read(path);
  }

  /** How many files are waiting, which every answer reports. */
  pendingCount(): number {
    return this.cache.pendingCount();
  }

  /**
   * Embeds everything queued.
   *
   * Only one drain runs at a time; asking again while one is in progress joins
   * the run already happening rather than embedding the same files twice.
   */
  drain(): Promise<void> {
    if (this.running) return this.running;

    this.running = this.drainOnce().finally(() => {
      this.running = null;
    });

    return this.running;
  }

  /** Asks an in-progress drain to stop at the next file boundary. */
  stop(): void {
    this.stopRequested = true;
    // Whatever was waiting to be embedded will be read again if the queue is
    // ever resumed, so there is no reason to keep holding it.
    this.held.clear();
  }

  private async drainOnce(): Promise<void> {
    this.stopRequested = false;

    // The queue itself holds only names. A file's text enters memory when its
    // turn comes and leaves again before the next one is opened, so a project
    // of any size costs one file at a time rather than all of them.
    for (const path of this.cache.dirtyFiles()) {
      if (this.stopRequested) return;

      let file = this.takeContent(path);
      if (!file) {
        this.cache.removeFile(path);
        continue;
      }

      try {
        await this.embedFile(file);
      } finally {
        // Released here rather than left to the next iteration, so the text is
        // collectable while the model works on what came out of it.
        file = null;
      }
    }
  }

  /**
   * Brings one file up to date.
   *
   * Only blocks never seen before are embedded: code that merely moved, or that
   * already appears elsewhere in the project, is already in the cache under the
   * same content hash.
   */
  private async embedFile(file: SourceFile): Promise<void> {
    const chunks = this.chunkerFor().chunk(file.file, file.content);
    const contentHash = hashContent(file.content);

    const wanted = new Set<string>();
    for (const chunk of chunks) {
      wanted.add(chunk.normalizedHash);
    }

    const known = this.cache.knownVectorHashes(Array.from(wanted));

    const missing: Chunk[] = [];
    const seen = new Set<string>();
    for (const chunk of chunks) {
      if (seen.has(chunk.normalizedHash)) continue;
      seen.add(chunk.normalizedHash);
      if (!known.has(chunk.normalizedHash)) missing.push(chunk);
    }

    for (let start = 0; start < missing.length; start += EMBED_BATCH_SIZE) {
      if (this.stopRequested) return;

      const end = Math.min(start + EMBED_BATCH_SIZE, missing.length);
      const texts: string[] = [];
      for (let i = start; i < end; i++) {
        texts.push(missing[i].text);
      }

      const vectors = await this.embedder.embed(texts);
      const entries: { chunkHash: string; vector: Embedding }[] = [];
      for (let i = start; i < end; i++) {
        entries.push({ chunkHash: missing[i].normalizedHash, vector: vectors[i - start] });
        // The block's text has served its purpose. A large file's blocks
        // overlap heavily, so together they can hold several times the file
        // itself; dropping each as it is embedded keeps that bounded.
        missing[i].text = '';
      }
      this.cache.putVectors(entries);
    }

    this.cache.markClean(file.file, file.mtime, byteLengthOf(file.content), contentHash, Array.from(wanted));
  }
}

/**
 * How many blocks are embedded before results are written down.
 *
 * A file of a few thousand lines can produce hundreds of blocks; committing in
 * batches keeps the vectors held in memory bounded and means an interrupted run
 * keeps most of its work.
 */
const EMBED_BATCH_SIZE = 64;

/**
 * How many already-read files may wait in memory for the drain to reach them.
 *
 * Enough to cover the ordinary case — a handful of files changed since the last
 * question — without letting a first index of a large project hold the whole
 * codebase, which is the very thing the queue exists to avoid.
 */
const MAX_HELD_FILES = 32;

/** A file bigger than this is re-read rather than held, to bound what is kept. */
const MAX_HELD_FILE_BYTES = 256 * 1024;

/** Identifies a file's content, so an unchanged file is never re-embedded. */
export function hashContent(content: string): string {
  return createHash('sha256').update(content).digest('hex').substring(0, 32);
}

function byteLengthOf(content: string): number {
  return Buffer.byteLength(content, 'utf-8');
}
