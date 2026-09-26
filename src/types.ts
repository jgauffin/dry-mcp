/**
 * The vocabulary of duplication analysis, in business terms.
 */

/** A candidate block of code that might turn out to be duplicated. */
export interface Chunk {
  /** Project-relative path, forward slashes. */
  file: string;
  /** 1-based, inclusive. */
  startLine: number;
  /** 1-based, inclusive. */
  endLine: number;
  /** The source as written, for reporting back to the developer. */
  text: string;
  /**
   * Hash of the block with formatting and comments removed. Two blocks sharing
   * this hash are the same code however differently it happens to be laid out,
   * which is what lets identical copies be found without embedding them.
   */
  normalizedHash: string;
  /** Lines remaining after comments and blank lines are discarded. */
  significantLines: number;
}

/** One occurrence of a repeated block. */
export interface Occurrence {
  file: string;
  startLine: number;
  endLine: number;
  lines: number;
}

/** Why a group of repeated code was not counted against the developer. */
export interface SuppressionReason {
  rule: 'small-and-frequent' | 'widespread-idiom' | 'ignored-path' | 'ignored-content';
  explanation: string;
}

/** A set of code blocks that say the same thing in more than one place. */
export interface DuplicationCluster {
  /** Stable identifier so a caller can ask for the full source later. */
  id: string;
  /** Every place this code appears. */
  occurrences: Occurrence[];
  /** How many places that is. */
  frequency: number;
  /** Typical size of one copy, in significant lines. */
  medianLines: number;
  /**
   * Lines that would disappear if the copies were folded into one definition.
   * This is the cost of the duplication, not its total footprint.
   */
  removableLines: number;
  /** Ranking score combining size and frequency. Higher is worse. */
  severity: number;
  /** 1.0 for identical copies; below that for near-misses found by embedding. */
  similarity: number;
  /** How the copies differ from each other, when they are not identical. */
  matchType: 'identical' | 'near-identical';
  /**
   * How sure the finding is, so a caller can tell a certain duplication from a
   * plausible one. Near-miss matching is deliberately inclusive — missing a
   * large repeated block is worse than offering one that turns out to be a
   * coincidence — which only works if the caller is told which is which.
   */
  confidence: Confidence;
  /** A first line or two, so a caller can recognise the code without a lookup. */
  preview: string;
  /** Present only when the cluster was demoted out of the main ranking. */
  suppressed?: SuppressionReason;
}

/** How the index is doing, reported alongside every answer. */
export interface IndexStatus {
  /** Files waiting to be embedded. Non-zero means the answer may be incomplete. */
  pendingFiles: number;
  filesIndexed: number;
  chunksCached: number;
  modelStatus: ModelStatus;
}

export type ModelStatus = 'ready' | 'not-installed' | 'incomplete';

/**
 * How far the index has got, so a caller can tell a complete answer from an
 * early one and decide whether to ask again.
 */
export interface IndexProgress {
  filesInScope: number;
  filesEmbedded: number;
  pendingFiles: number;
  /** 0 to 100. */
  percentComplete: number;
}

/** How many files one folder contributes to the analysis. */
export interface FolderSize {
  folder: string;
  files: number;
}

/**
 * What is being analysed, reported when the scope looks wider than the team
 * probably meant — the first run in a repository, where nothing has been
 * narrowed yet, is the case this exists for.
 */
export interface ScopeSummary {
  filesInScope: number;
  /** Where the scope is written down, whether or not the file exists yet. */
  configFile: string;
  /** False when there is no config file, so everything under the root is in scope. */
  configured: boolean;
  /** The folders contributing the most files, so an unwanted tree is easy to spot. */
  largestFolders: FolderSize[];
  /** Repositories of their own, left out of the analysis. */
  skippedNestedRepositories?: string[];
}

/**
 * How much to trust a finding.
 *
 * - `certain`   the blocks are the same code once formatting and comments are
 *               set aside. Not a judgement call.
 * - `high`      strongly alike; in practice a copy with renames or a small
 *               edit. Worth acting on without much checking.
 * - `moderate`  alike enough to be worth a look, but read the code first — a
 *               shared shape can score here without shared meaning.
 * - `low`       only loosely alike. Included so that large, frequently repeated
 *               blocks are not missed; expect some of these to be coincidence.
 */
export type Confidence = 'certain' | 'high' | 'moderate' | 'low';

/** What each confidence level means, sent with the results so it need not be guessed. */
export const CONFIDENCE_GUIDE: Record<Confidence, string> = {
  certain: 'Identical code once formatting and comments are set aside.',
  high: 'Almost certainly a copy, typically with renamed identifiers or a small edit.',
  moderate: 'Probably related, but read the code before acting — shared structure can score here.',
  low: 'Loosely similar. Reported so large repeated blocks are not missed; verify before acting.',
};

/** A vector, always unit length so similarity is a plain dot product. */
export type Embedding = Float32Array;
