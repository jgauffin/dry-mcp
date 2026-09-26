import type { DuplicationCluster, Occurrence } from '../types.js';

/**
 * Decides which duplications actually cost the team something.
 *
 * The ranking answers a specific question: if we only had time to fix one of
 * these, which one? A large block copied a few times hurts more than a tiny
 * fragment repeated everywhere, because every copy of the large block is a
 * place a future change has to be repeated and can be forgotten.
 */

/**
 * How much of the duplication could actually be deleted.
 *
 * One copy always has to stay, so the waste is every copy after the first.
 */
export function removableLines(medianLines: number, frequency: number): number {
  return medianLines * (frequency - 1);
}

/**
 * How much heavier a long block counts than a short one.
 *
 * Above one, size grows faster than linearly, which is what lets a large block
 * outrank a small one that occurs far more often. It reflects real cost: a
 * sixty-line block is more than twenty times harder to keep consistent than a
 * three-line one, because understanding it, spotting that a copy has drifted,
 * and changing every copy correctly all get disproportionately harder as it
 * grows.
 */
const SIZE_WEIGHT = 1.5;

/**
 * The cost of a duplication, used to rank it against the others.
 *
 * Size dominates and frequency merely modulates. A sixty-line block copied four
 * times must outrank a three-line fragment repeated forty times, because the
 * former is a genuine maintenance liability while the latter is almost always
 * an idiom. Frequency enters logarithmically so that going from two copies to
 * four matters much more than going from thirty to forty — by then it is a
 * pattern, not an incident.
 */
export function severityOf(medianLines: number, frequency: number): number {
  if (frequency < 2 || medianLines <= 0) return 0;

  const copiesWorthRemoving = frequency - 1;
  return round(Math.pow(medianLines, SIZE_WEIGHT) * Math.log2(1 + copiesWorthRemoving));
}

/** The typical size of one copy, resistant to a single unusual occurrence. */
export function medianLinesOf(occurrences: Occurrence[]): number {
  if (occurrences.length === 0) return 0;

  const sizes: number[] = [];
  for (const occurrence of occurrences) {
    sizes.push(occurrence.lines);
  }
  sizes.sort((a, b) => a - b);

  const middle = Math.floor(sizes.length / 2);
  if (sizes.length % 2 === 1) return sizes[middle];
  return Math.round((sizes[middle - 1] + sizes[middle]) / 2);
}

/** Worst first, so the caller reads the expensive problems before the cheap ones. */
export function rankBySeverity(clusters: DuplicationCluster[]): DuplicationCluster[] {
  const ranked = [...clusters];
  ranked.sort((a, b) => b.severity - a.severity || b.removableLines - a.removableLines);
  return ranked;
}

/**
 * Most widespread first. A separate view because "what is copied most often"
 * is a different question from "what costs us most", and both get asked.
 */
export function rankByFrequency(clusters: DuplicationCluster[]): DuplicationCluster[] {
  const ranked = [...clusters];
  ranked.sort((a, b) => b.frequency - a.frequency || b.severity - a.severity);
  return ranked;
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}
