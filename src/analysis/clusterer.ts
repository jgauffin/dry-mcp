import { createHash } from 'crypto';
import type {
  Chunk,
  Confidence,
  DuplicationCluster,
  Embedding,
  Occurrence,
} from '../types.js';
import { cosineSimilarity } from '../embedding/embedder.js';
import { medianLinesOf, removableLines, severityOf } from './ranker.js';
import { alignment, contentLines, type LineTokens } from './line-alignment.js';

export interface ClusteringOptions {
  /** How alike two blocks must be to be called the same code. */
  similarityThreshold: number;
}

/**
 * Only compare blocks of roughly the same size. Two blocks whose lengths differ
 * by more than this are not the same code however similar the embeddings look,
 * and skipping them keeps the comparison affordable on a large codebase.
 */
const SIZE_TOLERANCE = 0.3;

/**
 * Block sizes at which the configured threshold applies unchanged, and at which
 * it has risen to its strictest.
 *
 * Long blocks of the same language resemble each other far more than short ones
 * do — two unrelated hundred-line TypeScript files share imports, brace style,
 * naming habits and control flow, and score highly on all of it. Measured
 * against this model, unrelated hundred-line blocks reach about 0.57 while
 * unrelated short ones stay near 0.22. One fixed threshold therefore cannot
 * serve both: set for short blocks it groups half the codebase, set for long
 * ones it misses renamed functions.
 */
const SIZE_SCALING_STARTS_AT = 25;
const SIZE_SCALING_ENDS_AT = 100;

/**
 * How much stricter the threshold becomes for the longest blocks.
 *
 * Takes the default 0.45 up to 0.8 at a hundred lines. Measured against this
 * model, unrelated hundred-line blocks reach 0.57 and a genuine copy of that
 * size — even one renamed throughout — scores 0.89, so 0.8 sits clearly between
 * them.
 */
const LARGEST_BLOCK_PENALTY = 0.35;

/**
 * The least share of lines two blocks must have in common, in order, to be
 * called the same code once their embeddings agree.
 *
 * Embeddings alone group code that only has the same shape: on a real project
 * three unrelated test suites and sixteen unrelated tool classes each formed a
 * "duplication", all rated high. A copy keeps its lines; shape-alike code does
 * not. Measured on that project, those pairs aligned at 0.21 at most and test
 * cases sharing only a fixture's vocabulary at 0.5 to 0.61, while real copies
 * aligned from 0.67 up — a function renamed throughout, and the same logic
 * rewritten in C#, both reach 0.8.
 */
export const MIN_ALIGNMENT = 0.65;

/** Line alignment at and above which a near-miss may be trusted moderately, and highly. */
const MODERATE_ALIGNMENT = 0.7;
const HIGH_ALIGNMENT = 0.75;

/**
 * How much of an occurrence a larger finding must cover for the occurrence to
 * count as already reported.
 */
const REDUNDANT_COVERAGE = 0.8;

/**
 * Groups the code blocks that say the same thing.
 *
 * Works in two passes for a practical reason: blocks that are textually
 * identical can be grouped by their hash for free, which is both exact and
 * removes the bulk of the candidates. Only what remains is compared by meaning,
 * which is where near-misses — a copy with a renamed variable or an extra
 * statement — are caught, and which costs real time.
 */
export class Clusterer {
  private options: ClusteringOptions;

  constructor(options: ClusteringOptions) {
    this.options = options;
  }

  /**
   * Finds every group of repeated code.
   *
   * Vectors may be missing for blocks whose files are still queued; those are
   * still grouped by exact match, so a partial index gives a partial answer
   * rather than none.
   */
  cluster(chunks: Chunk[], vectors: Map<string, Embedding>): DuplicationCluster[] {
    const identical = this.groupIdentical(chunks);
    const clusters: DuplicationCluster[] = [];

    for (const group of identical) {
      if (group.length > 1) {
        clusters.push(this.buildCluster(group, 1, 'identical'));
      }
    }

    // Each set of identical blocks now stands in for itself as a single
    // representative, so near-miss matching compares distinct shapes rather
    // than re-comparing copies it has already grouped.
    const representatives: Chunk[] = [];
    const membersOf = new Map<string, Chunk[]>();

    for (const group of identical) {
      representatives.push(group[0]);
      membersOf.set(group[0].normalizedHash, group);
    }

    const near = this.groupSimilar(representatives, vectors);

    for (const group of near) {
      const members: Chunk[] = [];
      for (const representative of group.chunks) {
        const expanded = membersOf.get(representative.normalizedHash) ?? [representative];
        members.push(...expanded);
      }
      clusters.push(
        this.buildCluster(members, group.similarity, 'near-identical', group.alignment)
      );
    }

    return this.withoutRedundantClusters(clusters);
  }

  /**
   * Drops findings that live inside, or almost entirely over, a larger finding.
   *
   * Copying a function also copies the loop inside it, so both are genuinely
   * repeated. Reporting them separately would have a developer fix the
   * function and find the inner finding already gone — the same work counted
   * twice. The largest containing block is the actionable one, so only that is
   * kept.
   *
   * Almost is enough: two nested blocks that start a line apart are windowed a
   * line apart, and each set of windows matches the same copy elsewhere. That
   * is one duplication, not one per set of windows.
   */
  private withoutRedundantClusters(clusters: DuplicationCluster[]): DuplicationCluster[] {
    const bySize = [...clusters];
    bySize.sort((a, b) => b.medianLines - a.medianLines);

    const kept: DuplicationCluster[] = [];

    for (const candidate of bySize) {
      let contained = false;
      for (const larger of kept) {
        if (this.isContainedIn(candidate, larger)) {
          contained = true;
          break;
        }
      }
      if (!contained) kept.push(candidate);
    }

    return kept;
  }

  /**
   * Whether every occurrence of one finding sits inside, or almost entirely
   * over, an occurrence of another. Requiring all of them matters: a block
   * repeated both inside a duplicated function and somewhere else entirely is
   * still worth its own finding.
   */
  private isContainedIn(inner: DuplicationCluster, outer: DuplicationCluster): boolean {
    for (const occurrence of inner.occurrences) {
      let covered = false;
      for (const candidate of outer.occurrences) {
        if (coverageOf(occurrence, candidate) >= REDUNDANT_COVERAGE) {
          covered = true;
          break;
        }
      }
      if (!covered) return false;
    }
    return true;
  }

  /** Blocks whose normalized text is byte-for-byte the same. */
  private groupIdentical(chunks: Chunk[]): Chunk[][] {
    const byHash = new Map<string, Chunk[]>();

    for (const chunk of chunks) {
      const existing = byHash.get(chunk.normalizedHash);
      if (existing) {
        existing.push(chunk);
      } else {
        byHash.set(chunk.normalizedHash, [chunk]);
      }
    }

    const groups: Chunk[][] = [];
    for (const group of byHash.values()) {
      groups.push(this.withoutOverlaps(group));
    }
    return groups;
  }

  /**
   * Drops occurrences that sit inside another occurrence in the same group.
   *
   * Windowing a long block produces overlapping candidates, and reporting a
   * block plus the window inside it as two copies of each other would be an
   * artefact of how the file was cut up, not a duplication anyone can fix.
   */
  private withoutOverlaps(group: Chunk[]): Chunk[] {
    // Largest first, so when two windows overlap the fuller one is the survivor
    // rather than whichever happened to come first.
    const bySize = [...group];
    bySize.sort((a, b) => b.significantLines - a.significantLines);

    const kept: Chunk[] = [];

    for (const candidate of bySize) {
      // Any shared line at all, not merely full containment: consecutive
      // windows of one long block overlap partially, and counting both would
      // report one duplication as two.
      if (!this.overlapsAny(kept, candidate)) kept.push(candidate);
    }

    kept.sort((a, b) => a.file.localeCompare(b.file) || a.startLine - b.startLine);
    return kept;
  }

  /**
   * Groups blocks that mean the same thing without being written identically.
   *
   * Greedy: the first block starts a group and every unassigned block close
   * enough to it joins. That is cheaper than true agglomerative clustering and
   * good enough here, because the threshold is high and the question is which
   * code to look at, not an exact taxonomy.
   */
  private groupSimilar(
    chunks: Chunk[],
    vectors: Map<string, Embedding>
  ): { chunks: Chunk[]; similarity: number; alignment: number }[] {
    const comparable: Chunk[] = [];
    for (const chunk of chunks) {
      if (vectors.has(chunk.normalizedHash)) comparable.push(chunk);
    }

    // Longest first, so a group forms around the most substantial block rather
    // than around whichever fragment happened to come first.
    comparable.sort((a, b) => b.significantLines - a.significantLines);

    const assigned = new Set<string>();
    const groups: { chunks: Chunk[]; similarity: number; alignment: number }[] = [];
    // Each block's lines are worked out once, however many seeds it is held
    // up against.
    const linesOf = new Map<string, LineTokens[]>();
    const lines = (chunk: Chunk): LineTokens[] => {
      let known = linesOf.get(chunk.normalizedHash);
      if (!known) {
        known = contentLines(chunk.text);
        linesOf.set(chunk.normalizedHash, known);
      }
      return known;
    };

    for (let i = 0; i < comparable.length; i++) {
      const seed = comparable[i];
      if (assigned.has(seed.normalizedHash)) continue;

      const members: Chunk[] = [seed];
      let lowest = 1;
      let lowestAlignment = 1;

      for (let j = i + 1; j < comparable.length; j++) {
        const candidate = comparable[j];
        if (assigned.has(candidate.normalizedHash)) continue;
        if (!this.comparableInSize(seed, candidate)) continue;
        // Against every member, not just the seed: windows of one long block
        // overlap each other, and admitting them all would report a single
        // duplication as though it had happened five times.
        if (this.overlapsAny(members, candidate)) continue;

        const similarity = cosineSimilarity(
          vectors.get(seed.normalizedHash)!,
          vectors.get(candidate.normalizedHash)!
        );

        if (similarity < this.thresholdFor(seed, candidate)) continue;

        // Checked only once the embeddings agree, which keeps it to the few
        // pairs worth the cost.
        const aligned = alignment(lines(seed), lines(candidate));
        if (aligned < MIN_ALIGNMENT) continue;

        members.push(candidate);
        assigned.add(candidate.normalizedHash);
        if (similarity < lowest) lowest = similarity;
        if (aligned < lowestAlignment) lowestAlignment = aligned;
      }

      if (members.length > 1) {
        assigned.add(seed.normalizedHash);
        groups.push({
          chunks: members,
          similarity: round(lowest),
          alignment: round(lowestAlignment),
        });
      }
    }

    return groups;
  }

  /**
   * How alike these two particular blocks must be to count as the same code.
   *
   * The configured threshold is the one for short blocks. Longer blocks have to
   * clear a higher bar, because length alone makes unrelated code look similar:
   * without this a hundred-line window matches almost any other hundred-line
   * window in the same language, and the report fills with pairs that share
   * nothing but their size and syntax.
   */
  private thresholdFor(first: Chunk, second: Chunk): number {
    const base = this.options.similarityThreshold;
    const lines = Math.min(first.significantLines, second.significantLines);

    if (lines <= SIZE_SCALING_STARTS_AT) return base;

    const span = SIZE_SCALING_ENDS_AT - SIZE_SCALING_STARTS_AT;
    const howFar = Math.min(1, (lines - SIZE_SCALING_STARTS_AT) / span);

    return Math.min(0.98, base + howFar * LARGEST_BLOCK_PENALTY);
  }

  /** Whether two blocks are close enough in length to be the same code. */
  private comparableInSize(first: Chunk, second: Chunk): boolean {
    const larger = Math.max(first.significantLines, second.significantLines);
    const smaller = Math.min(first.significantLines, second.significantLines);
    if (larger === 0) return false;
    return smaller / larger >= 1 - SIZE_TOLERANCE;
  }

  /**
   * Whether two blocks cover any of the same lines of the same file.
   *
   * Windows are judged by the whole block they were cut from. Two windows of
   * one long block can be many lines apart and still be the same code — written
   * by the same hand in the same style, they score as alike as a real copy —
   * and the same goes for windows of two blocks nested one inside the other.
   */
  private overlaps(first: Chunk, second: Chunk): boolean {
    if (first.file !== second.file) return false;
    const firstStart = first.blockStartLine ?? first.startLine;
    const firstEnd = first.blockEndLine ?? first.endLine;
    const secondStart = second.blockStartLine ?? second.startLine;
    const secondEnd = second.blockEndLine ?? second.endLine;
    return firstStart <= secondEnd && secondStart <= firstEnd;
  }

  /** Whether a block shares lines with any block already in the group. */
  private overlapsAny(members: Chunk[], candidate: Chunk): boolean {
    for (const member of members) {
      if (this.overlaps(member, candidate)) return true;
    }
    return false;
  }

  private buildCluster(
    chunks: Chunk[],
    similarity: number,
    matchType: 'identical' | 'near-identical',
    aligned?: number
  ): DuplicationCluster {
    const occurrences: Occurrence[] = [];
    for (const chunk of chunks) {
      occurrences.push({
        file: chunk.file,
        startLine: chunk.startLine,
        endLine: chunk.endLine,
        lines: chunk.significantLines,
      });
    }

    occurrences.sort((a, b) => a.file.localeCompare(b.file) || a.startLine - b.startLine);

    const frequency = occurrences.length;
    const medianLines = medianLinesOf(occurrences);

    const cluster: DuplicationCluster = {
      id: identityOf(occurrences),
      occurrences,
      frequency,
      medianLines,
      removableLines: removableLines(medianLines, frequency),
      severity: severityOf(medianLines, frequency),
      similarity,
      matchType,
      confidence: confidenceOf(similarity, medianLines, matchType, aligned),
      preview: previewOf(chunks[0]),
    };
    if (aligned !== undefined) cluster.alignment = aligned;
    return cluster;
  }
}

/**
 * How much to trust a finding.
 *
 * Similarity alone is not enough, because the same score means different things
 * at different sizes: twenty lines agreeing closely is hard to do by accident,
 * while six lines can agree that well simply by being an ordinary loop. Size is
 * therefore allowed to raise or lower the verdict a step.
 */
function confidenceOf(
  similarity: number,
  medianLines: number,
  matchType: 'identical' | 'near-identical',
  aligned = 1
): Confidence {
  if (matchType === 'identical') return 'certain';

  const levels: Confidence[] = ['low', 'moderate', 'high'];

  // Judged against what a match of this size is worth. Long blocks of the same
  // language score highly on syntax alone, so 0.8 between two hundred-line
  // windows says much less than 0.8 between two short functions.
  const expected = expectedSimilarityFor(medianLines);
  const margin = similarity - expected;

  let level = margin >= 0.25 ? 2 : margin >= 0.1 ? 1 : 0;

  // A very short block can agree closely by being ordinary, so it is trusted
  // one step less whatever it scored.
  if (medianLines <= 8 && level > 0) level--;

  // However alike the embeddings say the code reads, a finding is only as
  // trustworthy as the lines it actually shares: embeddings score shape, this
  // scores copying.
  const byLines = aligned >= HIGH_ALIGNMENT ? 2 : aligned >= MODERATE_ALIGNMENT ? 1 : 0;

  return levels[Math.min(level, byLines)];
}

/**
 * How much of one occurrence another occurrence covers, from 0 to 1. Different
 * files never cover each other.
 */
function coverageOf(occurrence: Occurrence, by: Occurrence): number {
  if (occurrence.file !== by.file) return 0;

  const start = Math.max(occurrence.startLine, by.startLine);
  const end = Math.min(occurrence.endLine, by.endLine);
  if (end < start) return 0;

  return (end - start + 1) / (occurrence.endLine - occurrence.startLine + 1);
}

/**
 * Roughly what two unrelated blocks of this size score against each other.
 *
 * Measured against the model: unrelated short blocks sit near 0.2, unrelated
 * hundred-line blocks near 0.57. A finding is only interesting to the extent it
 * beats this baseline.
 */
function expectedSimilarityFor(lines: number): number {
  const shortBlockBaseline = 0.2;
  const longBlockBaseline = 0.57;

  if (lines <= 25) return shortBlockBaseline;
  if (lines >= 100) return longBlockBaseline;

  const howFar = (lines - 25) / 75;
  return shortBlockBaseline + howFar * (longBlockBaseline - shortBlockBaseline);
}

/**
 * A stable name for a cluster, so a caller can come back and ask for its full
 * source in a later request.
 */
function identityOf(occurrences: Occurrence[]): string {
  const parts: string[] = [];
  for (const occurrence of occurrences) {
    parts.push(`${occurrence.file}:${occurrence.startLine}-${occurrence.endLine}`);
  }
  return createHash('sha256').update(parts.join('|')).digest('hex').substring(0, 12);
}

/** Enough of the code to recognise it without fetching the whole block. */
function previewOf(chunk: Chunk): string {
  const lines = chunk.text.split('\n');
  const meaningful: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.length > 0) meaningful.push(trimmed);
    if (meaningful.length === 2) break;
  }

  const preview = meaningful.join(' ');
  return preview.length > 120 ? `${preview.substring(0, 120)}...` : preview;
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
