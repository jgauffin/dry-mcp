import type { DuplicationCluster } from '../types.js';

/** A finding as it was handed out, with the files it was derived from. */
export interface RememberedFinding {
  cluster: DuplicationCluster;
  /** Content hash per file involved, so a finding is never served against edited code. */
  contentHashes: Map<string, string>;
}

/**
 * How many findings are remembered before the oldest are let go.
 *
 * Generous enough to cover a session of questions about one project, while
 * keeping what is held to a bounded set of small records. Forgetting one only
 * costs a recomputed answer.
 */
const DEFAULT_LIMIT = 2000;

/**
 * Remembers the findings already reported, so a caller can come back and ask for
 * the source of one.
 *
 * A finding's id names the exact set of places its code was found, and that set
 * depends both on the question asked — how alike blocks must be, how long they
 * must be — and on how much of the project has been embedded so far. Looking an
 * id up by grouping the project again therefore fails to find perfectly good
 * findings, because the second grouping is not the one that produced the id.
 */
export class ReportedFindings {
  private findings = new Map<string, RememberedFinding>();
  private limit: number;

  constructor(limit: number = DEFAULT_LIMIT) {
    this.limit = limit;
  }

  /** Notes what an answer handed out, along with the state of the code behind it. */
  remember(clusters: readonly DuplicationCluster[], contentHashes: Map<string, string>): void {
    for (const cluster of clusters) {
      const involved = new Map<string, string>();
      for (const occurrence of cluster.occurrences) {
        const hash = contentHashes.get(occurrence.file);
        if (hash !== undefined) involved.set(occurrence.file, hash);
      }

      // Re-reporting a finding makes it recent again, so what a caller is
      // actively working through is the last thing to be forgotten.
      this.findings.delete(cluster.id);
      this.findings.set(cluster.id, { cluster, contentHashes: involved });
    }

    this.forgetOldest();
  }

  find(clusterId: string): RememberedFinding | null {
    return this.findings.get(clusterId) ?? null;
  }

  private forgetOldest(): void {
    while (this.findings.size > this.limit) {
      const oldest = this.findings.keys().next();
      if (oldest.done) return;
      this.findings.delete(oldest.value);
    }
  }
}
