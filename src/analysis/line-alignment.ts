import { stripComments } from '../chunking/normalizer.js';

/**
 * Whether two blocks keep the same lines in the same order, which is what a
 * copy does and what code that merely has the same shape does not.
 *
 * Embeddings answer "does this read alike", and on real code that is not the
 * same question as "was this copied". Two unrelated test suites share their
 * describe/it/expect scaffolding and score as highly as a genuine copy; so do
 * a dozen tool classes that each have an async execute method. What none of
 * them share is the lines themselves. A copy, even one with renamed variables
 * or an inserted statement, keeps most of its lines and keeps them in order.
 *
 * Like the rest of the analysis this works on words rather than a grammar, so
 * it covers every language without knowing any of them.
 */

/** A word, an identifier, or a number. Punctuation says nothing about whether code was copied. */
const TOKEN = /[A-Za-z_][A-Za-z0-9_]*|\d+/g;

/**
 * How much of a line two lines must share to be the same line.
 *
 * Half lets a line survive one renamed identifier — `let sum = 0` and
 * `let acc = 0` agree — while `expect(order.status).toBe('cancelled')` and
 * `expect(plan.steps).toHaveLength(3)` do not, despite having the same shape.
 */
const SAME_LINE = 0.5;

/** One line of a block, in the two forms lines are compared in. */
export interface LineTokens {
  /** The words on the line. */
  words: Set<string>;
  /**
   * The line with every word replaced by how far back the same word was last
   * used in the block, or null when no word on it had been used before.
   *
   * Renaming a variable everywhere changes every line it appears on, but not
   * where it appears: a copy renamed throughout has exactly the same distances
   * as the original. Code that is only shaped alike uses its words at
   * different distances. A line of nothing but first uses has no distances to
   * compare, and would match any other line of as many fresh words.
   */
  reuse: string | null;
}

/**
 * The lines of a block that carry words.
 *
 * Words inside strings are kept on purpose: two `it('...')` lines are the same
 * scaffolding but different tests, and the test names are what says so. Lines
 * of pure punctuation — closing braces, mostly — are dropped, since every block
 * has them and they would line up between any two.
 */
export function contentLines(text: string): LineTokens[] {
  const lines: LineTokens[] = [];
  const lastSeen = new Map<string, number>();
  let position = 0;

  for (const line of stripComments(text).split('\n')) {
    const words = line.match(TOKEN);
    if (!words) continue;

    const distances: number[] = [];
    let reused = false;

    for (const word of words) {
      const previous = lastSeen.get(word);
      if (previous === undefined) {
        distances.push(0);
      } else {
        distances.push(position - previous);
        reused = true;
      }
      lastSeen.set(word, position);
      position++;
    }

    lines.push({ words: new Set(words), reuse: reused ? distances.join(',') : null });
  }

  return lines;
}

/**
 * The share of the shorter block's lines that line up with the other block, in
 * order. 1 for a copy, near 0 for code that only shares a shape.
 *
 * Measured against the shorter block so that a copy with statements added still
 * counts as a copy of the original.
 */
export function alignment(first: LineTokens[], second: LineTokens[]): number {
  const shorter = Math.min(first.length, second.length);
  if (shorter === 0) return 0;

  // Longest common subsequence, keeping only the previous row: blocks are at
  // most a window long, so this is a few thousand comparisons per pair.
  let previous = new Array<number>(second.length + 1).fill(0);
  let current = new Array<number>(second.length + 1).fill(0);

  for (let i = 1; i <= first.length; i++) {
    for (let j = 1; j <= second.length; j++) {
      current[j] = sameLine(first[i - 1], second[j - 1])
        ? previous[j - 1] + 1
        : Math.max(previous[j], current[j - 1]);
    }
    [previous, current] = [current, previous];
  }

  return previous[second.length] / shorter;
}

/**
 * Whether two lines are the same line: most of the same words, or the same
 * words renamed.
 */
function sameLine(first: LineTokens, second: LineTokens): boolean {
  if (first.reuse !== null && first.reuse === second.reuse) return true;

  let shared = 0;
  for (const word of first.words) {
    if (second.words.has(word)) shared++;
  }

  const union = first.words.size + second.words.size - shared;
  return union > 0 && shared / union >= SAME_LINE;
}
