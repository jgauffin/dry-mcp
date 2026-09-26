import { minimatch } from 'minimatch';
import type { DuplicationCluster, SuppressionReason } from '../types.js';
import type { IdiomSettings } from '../config.js';

/**
 * Separates duplication worth fixing from repetition that is simply how the
 * language is written.
 *
 * Every codebase repeats itself harmlessly: property accessors, guard clauses,
 * null checks, constructor assignments. Reporting those alongside real
 * copy-paste teaches developers that the report is noise, and then the real
 * findings go unread too. Nothing is deleted here — demoted clusters are still
 * returned, with the reason, so the judgement can be checked and the rules
 * tuned.
 */
export class IdiomFilter {
  private settings: IdiomSettings;
  private ignoredPaths: string[];
  private contentPatterns: RegExp[];

  constructor(settings: IdiomSettings, ignoredPaths: string[] = []) {
    this.settings = settings;
    this.ignoredPaths = ignoredPaths;
    this.contentPatterns = compilePatterns(settings.ignorePatterns);
  }

  /**
   * Marks the clusters that are idioms rather than defects, leaving the rest
   * untouched.
   */
  apply(clusters: DuplicationCluster[]): DuplicationCluster[] {
    const judged: DuplicationCluster[] = [];

    for (const cluster of clusters) {
      const reason = this.reasonToSuppress(cluster);
      if (reason) {
        judged.push({ ...cluster, suppressed: reason });
      } else {
        judged.push(cluster);
      }
    }

    return judged;
  }

  /**
   * Why this cluster should not count against the team, or null when it should.
   */
  private reasonToSuppress(cluster: DuplicationCluster): SuppressionReason | null {
    const ignoredPath = this.matchIgnoredPath(cluster);
    if (ignoredPath) return ignoredPath;

    const ignoredContent = this.matchIgnoredContent(cluster);
    if (ignoredContent) return ignoredContent;

    if (this.isSmallAndFrequent(cluster)) {
      return {
        rule: 'small-and-frequent',
        explanation:
          `${cluster.medianLines} lines repeated ${cluster.frequency} times reads as a ` +
          `language idiom rather than copied logic. Raise idiom.minOccurrences or ` +
          `lower idiom.maxLines to see it ranked.`,
      };
    }

    if (this.isWidespreadIdiom(cluster)) {
      return {
        rule: 'widespread-idiom',
        explanation:
          `Appears in ${this.directoriesOf(cluster).size} separate directories, which ` +
          `suggests a shared convention rather than one copy-paste incident.`,
      };
    }

    return null;
  }

  /**
   * Short blocks that show up everywhere. The two thresholds have to be met
   * together: a short block copied twice is still worth a look, and a long
   * block is worth reporting no matter how often it appears.
   */
  private isSmallAndFrequent(cluster: DuplicationCluster): boolean {
    return (
      cluster.medianLines <= this.settings.maxLines &&
      cluster.frequency >= this.settings.minOccurrences
    );
  }

  /**
   * Code spread thinly across the whole tree.
   *
   * One team copying a block between two neighbouring files is an accident
   * worth fixing. The same shape appearing once in each of a dozen unrelated
   * folders is the codebase's house style, and no single owner could remove it.
   */
  private isWidespreadIdiom(cluster: DuplicationCluster): boolean {
    if (cluster.medianLines > this.settings.maxLines * 2) return false;

    const directories = this.directoriesOf(cluster);
    const spreadThinly = directories.size >= Math.max(4, cluster.frequency * 0.8);
    return spreadThinly && cluster.frequency >= this.settings.minOccurrences;
  }

  private directoriesOf(cluster: DuplicationCluster): Set<string> {
    const directories = new Set<string>();
    for (const occurrence of cluster.occurrences) {
      const lastSlash = occurrence.file.lastIndexOf('/');
      directories.add(lastSlash === -1 ? '.' : occurrence.file.substring(0, lastSlash));
    }
    return directories;
  }

  /**
   * Whether every copy sits somewhere the team has said not to report, such as
   * generated code or migrations. All of them must match: a duplicate that
   * straddles generated and hand-written code is still a real finding.
   */
  private matchIgnoredPath(cluster: DuplicationCluster): SuppressionReason | null {
    if (this.ignoredPaths.length === 0 || cluster.occurrences.length === 0) return null;

    for (const occurrence of cluster.occurrences) {
      if (!this.isIgnoredPath(occurrence.file)) return null;
    }

    return {
      rule: 'ignored-path',
      explanation: 'Every occurrence lies under a path excluded by configuration.',
    };
  }

  private isIgnoredPath(file: string): boolean {
    for (const pattern of this.ignoredPaths) {
      if (minimatch(file, pattern, { dot: true })) return true;
    }
    return false;
  }

  /** Whether the code itself matches a shape the team has chosen to accept. */
  private matchIgnoredContent(cluster: DuplicationCluster): SuppressionReason | null {
    for (const pattern of this.contentPatterns) {
      if (pattern.test(cluster.preview)) {
        return {
          rule: 'ignored-content',
          explanation: `Matches the configured ignore pattern /${pattern.source}/.`,
        };
      }
    }
    return null;
  }
}

/**
 * Turns configured patterns into matchers, skipping any that are malformed.
 *
 * A typo in one pattern must not take the whole analysis down; the rest of the
 * rules still apply and the developer simply keeps seeing that one finding.
 */
function compilePatterns(patterns: string[]): RegExp[] {
  const compiled: RegExp[] = [];

  for (const pattern of patterns) {
    try {
      compiled.push(new RegExp(pattern, 'm'));
    } catch {
      // An unusable pattern is ignored rather than fatal.
    }
  }

  return compiled;
}
