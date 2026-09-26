import { minimatch } from 'minimatch';
import * as fs from 'fs';
import * as path from 'path';
import {
  CONFIDENCE_GUIDE,
  type Chunk,
  type Confidence,
  type DuplicationCluster,
  type Embedding,
  type IndexStatus,
} from '../types.js';
import { readConfigFile, type DuplicationConfig } from '../config.js';
import { Chunker } from '../chunking/chunker.js';
import { Clusterer } from './clusterer.js';
import { IdiomFilter } from './idiom-filter.js';
import { rankBySeverity, rankByFrequency } from './ranker.js';
import { EmbeddingCache } from '../cache/embedding-cache.js';
import { ModelStore } from '../embedding/model-store.js';
import { LocalEmbedder, type Embedder } from '../embedding/embedder.js';
import { GenerationQueue, type FileScan, type SourceFile } from '../embedding/queue.js';
import { normalizePath } from '../paths.js';

/** The longest block compared as a single unit, before windowing takes over. */
const MAX_BLOCK_LINES = 120;

/** File extensions worth reading. Deliberately broad — the analysis is language-agnostic. */
const SOURCE_EXTENSIONS = [
  '.ts', '.tsx', '.js', '.jsx', '.mts', '.cts', '.mjs', '.cjs',
  '.cs', '.java', '.kt', '.go', '.rs', '.rb', '.php', '.py',
  '.c', '.h', '.cpp', '.hpp', '.cc', '.m', '.swift', '.scala',
  '.sql', '.sh', '.ps1', '.vb', '.css', '.scss', '.less',
];

/** Directories never worth walking into. */
const SKIP_DIRECTORIES = ['node_modules', 'dist', 'build', 'coverage', 'bin', 'obj', 'vendor'];

/** The largest file worth reading, so one generated blob cannot stall a scan. */
const MAX_FILE_BYTES = 2 * 1024 * 1024;

export interface DuplicationQuery {
  topN?: number;
  minLines?: number;
  orderBy?: 'severity' | 'frequency';
  includeSuppressed?: boolean;
  /** Leave out anything less certain than this. Defaults to reporting everything. */
  minConfidence?: Confidence;
  /**
   * How alike two blocks must be to count as the same code, overriding the
   * project setting for this question only.
   */
  similarityThreshold?: number;
}

export interface DuplicationReport {
  status: IndexStatus;
  /** Present only when the answer is limited or incomplete, so a clean run stays quiet. */
  notice?: string;
  analysisMode: 'heuristic';
  /**
   * What each confidence level means. Travels with the results so a caller can
   * weigh a finding without having to guess the scale.
   */
  confidenceGuide: Record<Confidence, string>;
  duplications: DuplicationCluster[];
  suppressedIdioms?: DuplicationCluster[];
  summary: {
    clustersFound: number;
    totalRemovableLines: number;
    filesInvolved: number;
    /** How the findings break down by how much they can be trusted. */
    byConfidence: Partial<Record<Confidence, number>>;
  };
}

/**
 * Answers the question the whole server exists for: where is the duplication
 * that is worth someone's afternoon?
 *
 * Holds the pieces together — reading the project, keeping embeddings current,
 * grouping repeated code, and deciding what is a real finding rather than an
 * idiom — and makes sure every answer says how complete it is.
 */
export class DuplicationService {
  private projectRoot: string;
  private config: DuplicationConfig;
  private chunker: Chunker;
  private clusterer: Clusterer;
  private idiomFilter: IdiomFilter;
  private cache: EmbeddingCache;
  private modelStore: ModelStore;
  private queue: GenerationQueue;
  private draining: Promise<void> | null = null;
  /** What the config file said last time it was read, to notice a rewrite. */
  private configFingerprint: string;

  constructor(
    projectRoot: string,
    config: DuplicationConfig,
    cache: EmbeddingCache,
    modelStore: ModelStore,
    embedder?: Embedder
  ) {
    this.projectRoot = normalizePath(projectRoot);
    this.config = config;
    this.configFingerprint = scopeFingerprintOf(config);
    this.modelStore = modelStore;
    this.cache = cache;
    this.chunker = new Chunker({ minLines: config.minLines, maxLines: MAX_BLOCK_LINES });
    this.clusterer = new Clusterer({ similarityThreshold: config.similarityThreshold });
    this.idiomFilter = new IdiomFilter(config.idiom, config.exclude);
    this.queue = new GenerationQueue(
      cache,
      () => this.chunker,
      embedder ?? new LocalEmbedder(modelStore),
      {
        list: () => this.listProject(),
        read: (file) => this.readSource(file),
      }
    );
  }

  /**
   * Brings the queue in line with the code on disk and with the rules the team
   * currently has in force.
   *
   * Cheap enough to do before every request: the walk only looks at names,
   * sizes and timestamps, and a file is opened only when one of those says it
   * may have changed.
   */
  refresh(): void {
    this.adoptConfigChanges();
    this.queue.scanForChanges({ maxFileBytes: MAX_FILE_BYTES });
  }

  /**
   * Picks up edits to the project's include and exclude rules.
   *
   * A developer who narrows the exclude list expects the newly covered folder
   * to be analysed, and one who widens it expects the excluded folder to stop
   * costing embedding work and stop appearing in answers. Neither happens if
   * the rules are only read when the server starts, since an editor session
   * outlives many config edits.
   */
  private adoptConfigChanges(): void {
    let reloaded: DuplicationConfig | null;
    try {
      reloaded = readConfigFile(this.projectRoot);
    } catch {
      // A half-typed config file is a moment in the developer's editing, not a
      // reason to stop answering. The rules in force stay in force.
      return;
    }

    // No file means the team never wrote settings down, so whatever this
    // service was constructed with is what they meant.
    if (!reloaded) return;

    const fingerprint = scopeFingerprintOf(reloaded);
    if (fingerprint === this.configFingerprint) return;

    this.configFingerprint = fingerprint;
    this.config = reloaded;
    this.chunker = new Chunker({ minLines: reloaded.minLines, maxLines: MAX_BLOCK_LINES });
    this.clusterer = new Clusterer({ similarityThreshold: reloaded.similarityThreshold });
    this.idiomFilter = new IdiomFilter(reloaded.idiom, reloaded.exclude);
  }

  /**
   * Starts embedding whatever is queued, without making the caller wait.
   *
   * A developer asking about duplication wants an answer now, from what is
   * ready; the queue count in the reply tells them how much of the project
   * that answer covers.
   */
  startEmbedding(): void {
    if (this.modelStore.status() !== 'ready') return;
    if (this.draining) return;

    this.draining = this.queue
      .drain()
      .catch(() => {
        // A failed run leaves the files queued, so the next request retries.
      })
      .finally(() => {
        this.draining = null;
      });
  }

  /** Waits for embedding to finish, for callers that want a complete answer. */
  async waitForEmbedding(): Promise<void> {
    this.startEmbedding();
    if (this.draining) await this.draining;
  }

  /** How complete the index is, reported with every answer. */
  status(): IndexStatus {
    return {
      pendingFiles: this.queue.pendingCount(),
      // Counted in the database rather than by loading every record: status is
      // asked for on every reply, and the records carry each file's whole
      // block list.
      filesIndexed: this.cache.fileCount(),
      chunksCached: this.cache.chunkCount(),
      modelStatus: this.modelStore.status(),
    };
  }

  /**
   * Finds and ranks the duplication in the project.
   *
   * When the model is missing this returns nothing rather than failing: the
   * caller is told what to install and can act, which an error would not
   * achieve.
   */
  analyze(query: DuplicationQuery = {}): DuplicationReport {
    const status = this.status();

    if (status.modelStatus !== 'ready') {
      return {
        status,
        notice: this.modelStore.explainUnavailable(),
        analysisMode: 'heuristic',
        confidenceGuide: CONFIDENCE_GUIDE,
        duplications: [],
        summary: {
          clustersFound: 0,
          totalRemovableLines: 0,
          filesInvolved: 0,
          byConfidence: {},
        },
      };
    }

    const chunks = this.collectChunks(query.minLines);
    const vectors = this.vectorsFor(chunks);
    const judged = this.idiomFilter.apply(
      this.clustererFor(query.similarityThreshold).cluster(chunks, vectors)
    );

    const reportable: DuplicationCluster[] = [];
    const suppressed: DuplicationCluster[] = [];
    for (const cluster of judged) {
      if (cluster.suppressed) {
        suppressed.push(cluster);
      } else {
        reportable.push(cluster);
      }
    }

    const trusted = this.atLeastAsSureAs(reportable, query.minConfidence);
    const ordered =
      query.orderBy === 'frequency' ? rankByFrequency(trusted) : rankBySeverity(trusted);
    const top = ordered.slice(0, query.topN ?? 20);

    const report: DuplicationReport = {
      status,
      analysisMode: 'heuristic',
      confidenceGuide: CONFIDENCE_GUIDE,
      duplications: top,
      summary: {
        clustersFound: trusted.length,
        totalRemovableLines: sumRemovable(trusted),
        filesInvolved: filesInvolvedIn(trusted),
        byConfidence: countByConfidence(trusted),
      },
    };

    const notice = this.noticeFor(status, suppressed.length);
    if (notice) report.notice = notice;
    if (query.includeSuppressed) report.suppressedIdioms = rankBySeverity(suppressed);

    return report;
  }

  /** The full source of one finding, for acting on it. */
  explain(clusterId: string): { cluster: DuplicationCluster; sources: { file: string; startLine: number; text: string }[] } | null {
    const chunks = this.collectChunks();
    const vectors = this.vectorsFor(chunks);
    const clusters = this.idiomFilter.apply(this.clusterer.cluster(chunks, vectors));

    const match = clusters.find((cluster) => cluster.id === clusterId);
    if (!match) return null;

    const sources: { file: string; startLine: number; text: string }[] = [];
    for (const occurrence of match.occurrences) {
      const content = this.readFile(occurrence.file);
      if (!content) continue;
      const lines = content.split('\n');
      sources.push({
        file: occurrence.file,
        startLine: occurrence.startLine,
        text: lines.slice(occurrence.startLine - 1, occurrence.endLine).join('\n'),
      });
    }

    return { cluster: match, sources };
  }

  /** Queues everything again, for when the caller wants a clean rebuild. */
  reindex(): void {
    this.cache.markAllDirty();
    this.refresh();
  }

  /** Discards vectors for code that no longer exists anywhere. */
  prune(): number {
    return this.cache.pruneOrphanedVectors();
  }

  dispose(): void {
    this.queue.stop();
    this.cache.close();
  }

  /**
   * Explains anything that limits how much the answer can be trusted: work
   * still queued, or findings hidden behind the idiom rules.
   */
  private noticeFor(status: IndexStatus, suppressedCount: number): string | undefined {
    const parts: string[] = [];

    if (status.pendingFiles > 0) {
      parts.push(
        `${status.pendingFiles} file(s) are queued for embedding, so recent changes may ` +
          `not be reflected yet. Ask again shortly for a complete picture.`
      );
    }

    if (suppressedCount > 0) {
      parts.push(
        `${suppressedCount} group(s) were treated as accepted idioms rather than ` +
          `duplication. Pass includeSuppressed to see them and why.`
      );
    }

    return parts.length > 0 ? parts.join(' ') : undefined;
  }

  /**
   * The grouping to use for one question.
   *
   * Reuses the project's own setting unless the caller asked for a different
   * one, which lets an agent widen the net when it is hunting for something and
   * narrow it when it wants only near-certain matches.
   */
  private clustererFor(similarityThreshold?: number): Clusterer {
    if (
      similarityThreshold === undefined ||
      similarityThreshold === this.config.similarityThreshold
    ) {
      return this.clusterer;
    }

    const bounded = Math.min(1, Math.max(0, similarityThreshold));
    return new Clusterer({ similarityThreshold: bounded });
  }

  /**
   * Keeps only the findings a caller is willing to trust.
   *
   * Everything is reported by default, because a large repeated block missed
   * entirely is worse than one the caller has to check. A caller that would
   * rather see less can say so.
   */
  private atLeastAsSureAs(
    clusters: DuplicationCluster[],
    minimum?: Confidence
  ): DuplicationCluster[] {
    if (!minimum) return clusters;

    const floor = CONFIDENCE_ORDER.indexOf(minimum);
    if (floor <= 0) return clusters;

    const kept: DuplicationCluster[] = [];
    for (const cluster of clusters) {
      if (CONFIDENCE_ORDER.indexOf(cluster.confidence) >= floor) kept.push(cluster);
    }
    return kept;
  }

  /**
   * Every block in the project worth comparing.
   *
   * Files are read one at a time and their text dropped once chunked, so what
   * survives the walk is the blocks themselves rather than a copy of the whole
   * codebase alongside them.
   */
  private collectChunks(minLines?: number): Chunk[] {
    const threshold = minLines ?? this.config.minLines;
    const chunks: Chunk[] = [];

    for (const scan of this.listProject()) {
      if (scan.size > MAX_FILE_BYTES) continue;

      const file = this.readSource(scan.file);
      if (!file) continue;

      for (const chunk of this.chunker.chunk(file.file, file.content)) {
        if (chunk.significantLines >= threshold) chunks.push(chunk);
      }
    }

    return chunks;
  }

  /** The vectors already computed for these blocks; misses are simply absent. */
  private vectorsFor(chunks: Chunk[]): Map<string, Embedding> {
    const hashes: string[] = [];
    const seen = new Set<string>();

    for (const chunk of chunks) {
      if (seen.has(chunk.normalizedHash)) continue;
      seen.add(chunk.normalizedHash);
      hashes.push(chunk.normalizedHash);
    }

    return this.cache.getVectors(hashes);
  }

  /**
   * Names every file in scope, honouring the configured include and exclude
   * rules, without opening any of them.
   *
   * The walk carries only names, sizes and timestamps, which is all that is
   * needed to decide what has changed — and is what makes it affordable to run
   * before every question on a project of any size.
   */
  private listProject(): FileScan[] {
    const scans: FileScan[] = [];
    this.walk(this.projectRoot, scans);
    return scans;
  }

  private walk(directory: string, collected: FileScan[]): void {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(directory, { withFileTypes: true });
    } catch {
      return;
    }

    for (const entry of entries) {
      const full = normalizePath(path.join(directory, entry.name));

      if (entry.isDirectory()) {
        if (SKIP_DIRECTORIES.includes(entry.name) || entry.name.startsWith('.')) continue;

        // Excluding a folder should save the walk as well as the reading. On a
        // large repository the directories a team excludes are usually the
        // biggest ones, so not descending into them is most of the saving.
        const relativeDirectory = normalizePath(path.relative(this.projectRoot, full));
        if (this.isExcluded(relativeDirectory) || this.isExcluded(`${relativeDirectory}/`)) {
          continue;
        }

        this.walk(full, collected);
        continue;
      }

      if (!SOURCE_EXTENSIONS.includes(path.extname(entry.name).toLowerCase())) continue;

      const relative = normalizePath(path.relative(this.projectRoot, full));
      if (!this.isIncluded(relative)) continue;

      try {
        const stats = fs.statSync(full);
        collected.push({ file: relative, mtime: stats.mtimeMs, size: stats.size });
      } catch {
        // An unreadable file is skipped rather than failing the whole scan.
      }
    }
  }

  /** One file's text, read only when something actually needs it. */
  private readSource(relative: string): SourceFile | null {
    const full = path.join(this.projectRoot, relative);

    try {
      const stats = fs.statSync(full);
      if (stats.size > MAX_FILE_BYTES) return null;

      return {
        file: relative,
        content: fs.readFileSync(full, 'utf-8'),
        mtime: stats.mtimeMs,
      };
    } catch {
      return null;
    }
  }

  /**
   * Whether a file should be analysed at all.
   *
   * Excluded files are still read for context nowhere else, so the cheapest
   * correct thing is to leave them out of the index entirely.
   */
  private isIncluded(relative: string): boolean {
    if (this.isExcluded(relative)) return false;

    for (const pattern of this.config.include) {
      if (minimatch(relative, pattern, { dot: true })) return true;
    }

    return false;
  }

  private isExcluded(relative: string): boolean {
    for (const pattern of this.config.exclude) {
      if (minimatch(relative, pattern, { dot: true })) return true;
    }

    return false;
  }

  private readFile(relative: string): string | null {
    try {
      return fs.readFileSync(path.join(this.projectRoot, relative), 'utf-8');
    } catch {
      return null;
    }
  }
}

/**
 * Identifies the settings that decide what gets indexed and how.
 *
 * Only these matter to the queue: changing which files are in scope, or how
 * blocks are cut, changes what has to be embedded, while changing how findings
 * are ranked or reported does not.
 */
function scopeFingerprintOf(config: DuplicationConfig): string {
  return JSON.stringify({
    include: config.include,
    exclude: config.exclude,
    minLines: config.minLines,
    similarityThreshold: config.similarityThreshold,
    idiom: config.idiom,
  });
}

/** Confidence from least to most sure, for comparing one level against another. */
const CONFIDENCE_ORDER: Confidence[] = ['low', 'moderate', 'high', 'certain'];

/** How many findings there are at each level of confidence. */
function countByConfidence(clusters: DuplicationCluster[]): Partial<Record<Confidence, number>> {
  const counts: Partial<Record<Confidence, number>> = {};

  for (const cluster of clusters) {
    counts[cluster.confidence] = (counts[cluster.confidence] ?? 0) + 1;
  }

  return counts;
}

function sumRemovable(clusters: DuplicationCluster[]): number {
  let total = 0;
  for (const cluster of clusters) {
    total += cluster.removableLines;
  }
  return total;
}

function filesInvolvedIn(clusters: DuplicationCluster[]): number {
  const files = new Set<string>();
  for (const cluster of clusters) {
    for (const occurrence of cluster.occurrences) {
      files.add(occurrence.file);
    }
  }
  return files.size;
}
