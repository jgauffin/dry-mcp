import { minimatch } from 'minimatch';
import * as fs from 'fs';
import * as path from 'path';
import {
  CONFIDENCE_GUIDE,
  type Chunk,
  type Confidence,
  type DuplicationCluster,
  type Embedding,
  type FolderSize,
  type IndexProgress,
  type IndexStatus,
  type ScopeSummary,
} from '../types.js';
import { CONFIG_FILE_NAME, readConfigFile, type DuplicationConfig } from '../config.js';
import { Chunker } from '../chunking/chunker.js';
import { Clusterer } from './clusterer.js';
import { IdiomFilter } from './idiom-filter.js';
import { rankBySeverity, rankByFrequency } from './ranker.js';
import { EmbeddingCache } from '../cache/embedding-cache.js';
import { ModelStore } from '../embedding/model-store.js';
import { LocalEmbedder, type Embedder } from '../embedding/embedder.js';
import {
  GenerationQueue,
  hashContent,
  type FileScan,
  type SourceFile,
} from '../embedding/queue.js';
import { IndexingPace } from '../embedding/pacer.js';
import { ReportedFindings, type RememberedFinding } from './reported-findings.js';
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

/**
 * A project of at most this many files is embedded in well under a minute, so
 * there is no early answer worth distinguishing from a complete one: it is
 * analysed as it stands and the queue count says the rest.
 */
const SMALL_PROJECT_FILES = 200;

/**
 * How much of a larger project must be embedded before findings are reported at
 * all.
 *
 * A ranking drawn from a fifth of a codebase is not an early version of the real
 * answer, it is a different answer: the worst duplication is most likely in the
 * part not yet read, and the ranking invites acting on whatever happens to be
 * indexed first. Below this share the reply carries progress instead, which
 * costs nothing and is honest about what is known.
 */
const USEFUL_INDEX_PERCENT = 50;

/** How many folders and skipped repositories a reply names before summarising. */
const MAX_PATHS_LISTED = 8;

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

/** Every block worth comparing, and the state of the files they came from. */
interface ProjectBlocks {
  chunks: Chunk[];
  /** Content hash per file, so a finding can later be told apart from an edited file. */
  contentHashes: Map<string, string>;
}

/** One copy's source, as returned when a finding is explained. */
interface OccurrenceSource {
  file: string;
  startLine: number;
  text: string;
}

/** What one walk of the project learned, kept so a reply can describe the scope. */
interface ScanTally {
  scans: FileScan[];
  /** Files per top-level folder, for naming the biggest contributors. */
  filesByFolder: Map<string, number>;
  /** Directories that are repositories of their own and were left out. */
  nestedRepositories: string[];
}

export interface DuplicationReport {
  status: IndexStatus;
  /** Present only when the answer is limited or incomplete, so a clean run stays quiet. */
  notice?: string;
  /** Present while the index is incomplete, so an early answer can be judged. */
  progress?: IndexProgress;
  /** Present when the scope looks wider than intended, with what to do about it. */
  scope?: ScopeSummary;
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
  /** Keeps background indexing from taking the machine the developer is working on. */
  private pace = new IndexingPace();
  private draining: Promise<void> | null = null;
  /** What has already been reported, so a finding can be looked up by its id. */
  private reported = new ReportedFindings();
  /** What the config file said last time it was read, to notice a rewrite. */
  private configFingerprint: string;
  /** What the most recent walk found, so describing the scope costs no extra walk. */
  private lastScan: ScanTally | null = null;

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
      },
      this.pace
    );
  }

  /**
   * Runs a request at full speed, and lets background indexing run at full speed
   * for as long as it lasts.
   *
   * Someone is waiting for a request, so pacing it would only make them wait
   * longer; between requests nobody is, which is when the pace applies.
   */
  whileAnswering<T>(work: () => Promise<T>): Promise<T> {
    return this.pace.whileAnswering(work);
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

    const progress = this.progress(status);
    if (this.tooEarlyToRank(progress)) return this.progressReport(status, progress);

    const blocks = this.readProject(query.minLines);
    const vectors = this.vectorsFor(blocks.chunks);
    const judged = this.idiomFilter.apply(
      this.clustererFor(query.similarityThreshold).cluster(blocks.chunks, vectors)
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

    const notice = this.noticeFor(progress, suppressed.length);
    if (notice) report.notice = notice;
    if (progress.pendingFiles > 0) report.progress = progress;

    const scope = this.scopeWorthReporting(progress);
    if (scope) report.scope = scope;
    if (query.includeSuppressed) report.suppressedIdioms = rankBySeverity(suppressed);

    // Only what this answer hands out: an id a caller never saw is not one they
    // can come back and ask about.
    this.reported.remember(report.duplications, blocks.contentHashes);
    if (report.suppressedIdioms) {
      this.reported.remember(report.suppressedIdioms, blocks.contentHashes);
    }

    return report;
  }

  /**
   * The full source of one finding, for acting on it.
   *
   * Answered from what was reported, because that is the only grouping the id
   * belongs to: the same project regrouped under a different threshold, a
   * different minimum block length, or simply with more of it embedded, puts the
   * code in different groups and so gives it different ids.
   */
  explain(
    clusterId: string
  ): { cluster: DuplicationCluster; sources: OccurrenceSource[] } | null {
    const remembered = this.reported.find(clusterId);
    if (remembered) {
      const sources = this.sourcesAsReported(remembered);
      if (sources) return { cluster: remembered.cluster, sources };
    }

    // Either nothing remembers this id — the server has restarted, or the answer
    // that carried it is long past — or the code has moved since, which makes the
    // remembered line ranges describe the wrong place. Group the project as it is
    // now and see whether the finding is still there.
    const blocks = this.readProject();
    const vectors = this.vectorsFor(blocks.chunks);
    const clusters = this.idiomFilter.apply(this.clusterer.cluster(blocks.chunks, vectors));

    const match = clusters.find((cluster) => cluster.id === clusterId);
    if (!match) return null;

    return { cluster: match, sources: this.sourcesOf(match, (file) => this.readFile(file)) };
  }

  /**
   * The source of a remembered finding, or nothing if any file behind it has
   * changed since — the line ranges then point somewhere else, and reporting
   * that as the duplicated code would be worse than not answering.
   */
  private sourcesAsReported(remembered: RememberedFinding): OccurrenceSource[] | null {
    const texts = new Map<string, string>();

    for (const [file, contentHash] of remembered.contentHashes) {
      const content = this.readFile(file);
      if (content === null || hashContent(content) !== contentHash) return null;
      texts.set(file, content);
    }

    return this.sourcesOf(remembered.cluster, (file) => texts.get(file) ?? null);
  }

  private sourcesOf(
    cluster: DuplicationCluster,
    textOf: (file: string) => string | null
  ): OccurrenceSource[] {
    const sources: OccurrenceSource[] = [];

    for (const occurrence of cluster.occurrences) {
      const content = textOf(occurrence.file);
      if (content === null) continue;
      const lines = content.split('\n');
      sources.push({
        file: occurrence.file,
        startLine: occurrence.startLine,
        text: lines.slice(occurrence.startLine - 1, occurrence.endLine).join('\n'),
      });
    }

    return sources;
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
  private noticeFor(progress: IndexProgress, suppressedCount: number): string | undefined {
    const parts: string[] = [];

    if (progress.pendingFiles > 0) {
      parts.push(
        `${progress.pendingFiles} file(s) are queued for embedding — ` +
          `${progress.percentComplete}% of ${progress.filesInScope} file(s) in scope are ` +
          `indexed — so this answer may not cover all of the project. Ask again shortly ` +
          `for a complete picture; embedding continues in the background either way.`
      );
    }

    if (suppressedCount > 0) {
      parts.push(
        `${suppressedCount} group(s) were treated as accepted idioms rather than ` +
          `duplication. Pass includeSuppressed to see them and why.`
      );
    }

    const scopeNotice = this.scopeNotice(progress);
    if (scopeNotice) parts.push(scopeNotice);

    return parts.length > 0 ? parts.join(' ') : undefined;
  }

  /** How far the index has got, which every incomplete answer carries. */
  progress(status: IndexStatus = this.status()): IndexProgress {
    // The walk knows the scope exactly; before one has run, what the cache has a
    // record of is the best available answer.
    const filesInScope = this.lastScan?.scans.length ?? status.filesIndexed;
    const embedded = Math.max(0, filesInScope - status.pendingFiles);

    return {
      filesInScope,
      filesEmbedded: embedded,
      pendingFiles: status.pendingFiles,
      percentComplete: filesInScope === 0 ? 100 : Math.floor((embedded / filesInScope) * 100),
    };
  }

  /**
   * Whether ranking the project now would say more about indexing order than
   * about the code. Small projects are always ranked: they finish before anyone
   * could ask twice.
   */
  private tooEarlyToRank(progress: IndexProgress): boolean {
    if (progress.pendingFiles === 0) return false;
    if (progress.filesInScope < SMALL_PROJECT_FILES) return false;
    return progress.percentComplete < USEFUL_INDEX_PERCENT;
  }

  /** Progress in place of findings, for a project still too thinly indexed to rank. */
  private progressReport(status: IndexStatus, progress: IndexProgress): DuplicationReport {
    const parts = [
      `Indexing this project: ${progress.percentComplete}% done ` +
        `(${progress.filesEmbedded} of ${progress.filesInScope} file(s) embedded, ` +
        `${progress.pendingFiles} queued). No findings yet — a ranking drawn from ` +
        `${progress.percentComplete}% of the code would mostly reflect which files were ` +
        `read first. Embedding runs in the background; ask again shortly, or call ` +
        `duplication_status to watch the queue.`,
    ];

    const scopeNotice = this.scopeNotice(progress);
    if (scopeNotice) parts.push(scopeNotice);

    const report: DuplicationReport = {
      status,
      notice: parts.join(' '),
      progress,
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

    const scope = this.scopeWorthReporting(progress);
    if (scope) report.scope = scope;

    return report;
  }

  /** What is in scope, described whenever the answer may be covering too much. */
  scopeSummary(filesInScope = this.progress().filesInScope): ScopeSummary {
    const summary: ScopeSummary = {
      filesInScope,
      configFile: CONFIG_FILE_NAME,
      configured: this.hasConfigFile(),
      largestFolders: this.largestFolders(),
    };

    const nested = this.lastScan?.nestedRepositories ?? [];
    if (nested.length > 0) summary.skippedNestedRepositories = nested;

    return summary;
  }

  /**
   * The scope, but only when something about it needs attention: nothing was
   * narrowed and the project is large, or whole repositories were left out. A
   * team that has written its rules down does not need them read back.
   */
  private scopeWorthReporting(progress: IndexProgress): ScopeSummary | undefined {
    if (!this.scopeNeedsAttention(progress)) return undefined;
    return this.scopeSummary(progress.filesInScope);
  }

  private scopeNeedsAttention(progress: IndexProgress): boolean {
    if ((this.lastScan?.nestedRepositories.length ?? 0) > 0) return true;
    return !this.hasConfigFile() && progress.filesInScope >= SMALL_PROJECT_FILES;
  }

  /**
   * What to tell a caller that can narrow the scope itself.
   *
   * The config file is plain JSON in the project root, so the calling agent can
   * write it without help, and the rules are re-read before the next question —
   * which is why this says what to write rather than who to ask.
   */
  private scopeNotice(progress: IndexProgress): string | undefined {
    if (!this.scopeNeedsAttention(progress)) return undefined;

    const parts: string[] = [];
    const nested = this.lastScan?.nestedRepositories ?? [];

    if (nested.length > 0) {
      parts.push(
        `${nested.length} nested repository/ies are left out, being other projects' code ` +
          `(submodule or vendored clone): ${listPaths(nested)}. Set ` +
          `"includeNestedRepositories": true in ${CONFIG_FILE_NAME} to analyse them too.`
      );
    }

    if (!this.hasConfigFile() && progress.filesInScope >= SMALL_PROJECT_FILES) {
      parts.push(
        `There is no ${CONFIG_FILE_NAME} in the project root, so every source file under ` +
          `it is in scope (${progress.filesInScope} files; largest folders: ` +
          `${describeFolders(this.largestFolders())}). If any of that is not code this team ` +
          `maintains, write the file yourself — ` +
          `{"exclude": ["vendor/**", "**/*.generated.*"], "include": ["src/**"]} — using ` +
          `globs on project-relative paths with forward slashes, where exclude wins over ` +
          `include. The next call picks the edit up; nothing needs restarting.`
      );
    }

    return parts.length > 0 ? parts.join(' ') : undefined;
  }

  private largestFolders(): FolderSize[] {
    const sizes: FolderSize[] = [];
    for (const [folder, files] of this.lastScan?.filesByFolder ?? []) {
      sizes.push({ folder, files });
    }

    sizes.sort((first, second) => second.files - first.files);
    return sizes.slice(0, MAX_PATHS_LISTED);
  }

  private hasConfigFile(): boolean {
    return fs.existsSync(path.join(this.projectRoot, CONFIG_FILE_NAME));
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
   * Every block in the project worth comparing, with a hash of each file it came
   * from.
   *
   * Files are read one at a time and their text dropped once chunked, so what
   * survives the walk is the blocks themselves rather than a copy of the whole
   * codebase alongside them.
   */
  private readProject(minLines?: number): ProjectBlocks {
    const threshold = minLines ?? this.config.minLines;
    const chunks: Chunk[] = [];
    const contentHashes = new Map<string, string>();

    for (const scan of this.listProject()) {
      if (scan.size > MAX_FILE_BYTES) continue;

      const file = this.readSource(scan.file);
      if (!file) continue;

      contentHashes.set(file.file, hashContent(file.content));

      for (const chunk of this.chunker.chunk(file.file, file.content)) {
        if (chunk.significantLines >= threshold) chunks.push(chunk);
      }
    }

    return { chunks, contentHashes };
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
    const tally: ScanTally = { scans: [], filesByFolder: new Map(), nestedRepositories: [] };
    this.walk(this.projectRoot, tally);
    this.lastScan = tally;
    return tally.scans;
  }

  private walk(directory: string, tally: ScanTally): void {
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

        // A submodule or a vendored clone is another project's code. Reported
        // rather than silently dropped, because a team that does own the
        // submodule needs to see why its code never appears.
        if (!this.config.includeNestedRepositories && isRepositoryRoot(full)) {
          tally.nestedRepositories.push(relativeDirectory);
          continue;
        }

        this.walk(full, tally);
        continue;
      }

      if (!SOURCE_EXTENSIONS.includes(path.extname(entry.name).toLowerCase())) continue;

      const relative = normalizePath(path.relative(this.projectRoot, full));
      if (!this.isIncluded(relative)) continue;

      try {
        const stats = fs.statSync(full);
        tally.scans.push({ file: relative, mtime: stats.mtimeMs, size: stats.size });
        const folder = topFolderOf(relative);
        tally.filesByFolder.set(folder, (tally.filesByFolder.get(folder) ?? 0) + 1);
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
    includeNestedRepositories: config.includeNestedRepositories,
    minLines: config.minLines,
    similarityThreshold: config.similarityThreshold,
    idiom: config.idiom,
  });
}

/**
 * Whether a directory is a repository in its own right.
 *
 * A submodule's `.git` is a file pointing at the parent's storage and a plain
 * clone's is a directory; either way its presence is what marks the boundary
 * between this project and somebody else's.
 */
function isRepositoryRoot(directory: string): boolean {
  return fs.existsSync(path.join(directory, '.git'));
}

/** The folder a file belongs to for reporting, or "." for the project root. */
function topFolderOf(relativeFile: string): string {
  const slash = relativeFile.indexOf('/');
  return slash === -1 ? '.' : relativeFile.substring(0, slash);
}

/** Paths for a notice, capped so one reply cannot list a thousand of them. */
function listPaths(paths: string[]): string {
  if (paths.length <= MAX_PATHS_LISTED) return paths.join(', ');

  const shown = paths.slice(0, MAX_PATHS_LISTED).join(', ');
  return `${shown} and ${paths.length - MAX_PATHS_LISTED} more`;
}

function describeFolders(folders: FolderSize[]): string {
  const described: string[] = [];
  for (const folder of folders) {
    described.push(`${folder.folder} (${folder.files})`);
  }
  return described.join(', ');
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
