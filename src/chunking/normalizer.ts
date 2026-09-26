import { createHash } from 'crypto';

/**
 * Reduces source text to just the code, so that two copies of the same logic
 * are recognised as the same however they happen to be written.
 *
 * None of this is a parser. Getting a comment boundary slightly wrong costs a
 * marginally noisier comparison, never a wrong answer, and that trade buys
 * support for every language the team writes without a grammar per language.
 */

/** Quote characters that begin a string in essentially every language. */
const QUOTES = new Set(['"', "'", '`']);

/**
 * Removes comments while leaving the line structure intact, so a block's
 * reported line numbers still point at the developer's actual source.
 *
 * String contents are tracked because a `//` inside a URL is code, and treating
 * it as a comment would silently delete the rest of a real line.
 */
export function stripComments(source: string): string {
  const lines = source.split('\n');
  const cleaned: string[] = [];
  let inBlockComment = false;
  // Carried across lines, because a template literal spans them and a "//"
  // inside one is text rather than the start of a comment.
  let quote: string | null = null;

  for (const line of lines) {
    let result = '';
    let index = 0;

    while (index < line.length) {
      const char = line[index];
      const next = line[index + 1];

      if (inBlockComment) {
        if (char === '*' && next === '/') {
          inBlockComment = false;
          index += 2;
          continue;
        }
        index++;
        continue;
      }

      if (quote) {
        // A backslash escapes the next character, so an escaped quote does not
        // end the string.
        if (char === '\\') {
          result += char + (next ?? '');
          index += 2;
          continue;
        }
        if (char === quote) quote = null;
        result += char;
        index++;
        continue;
      }

      if (QUOTES.has(char)) {
        quote = char;
        result += char;
        index++;
        continue;
      }

      if (char === '/' && next === '*') {
        inBlockComment = true;
        index += 2;
        continue;
      }

      if (isLineCommentStart(line, index)) break;

      result += char;
      index++;
    }

    // Only a template literal genuinely continues onto the next line.
    if (quote !== '`') quote = null;

    cleaned.push(result);
  }

  return cleaned.join('\n');
}

/**
 * Whether a comment that runs to the end of the line starts here.
 *
 * `--` is ambiguous: SQL and Lua use it for comments, while C-family languages
 * use it to decrement. Requiring whitespace in front keeps `count--` as code
 * while still catching `SELECT 1 -- note`.
 */
function isLineCommentStart(line: string, index: number): boolean {
  const char = line[index];
  const next = line[index + 1];

  if (char === '/' && next === '/') return true;
  if (char === '#') return true;
  if (char === '-' && next === '-') {
    const preceding = line[index - 1];
    return preceding === undefined || /\s/.test(preceding);
  }
  return false;
}

/**
 * The same text with comments gone and every string emptied of its contents.
 *
 * Only nesting punctuation that is really part of the code's structure should
 * count when deciding where a block begins and ends. A brace inside a message
 * or a regular expression is data, and letting it shift the nesting count makes
 * a block appear to run far past its actual end.
 */
export function structuralView(source: string): string {
  const withoutComments = stripComments(source);
  const lines = withoutComments.split('\n');
  const cleaned: string[] = [];

  // A template literal may span lines, and its contents keep hiding braces the
  // whole way. Ordinary quotes do not survive a line break, so only a backtick
  // is carried forward.
  let quote: string | null = null;
  // How deep inside `${ ... }` we are. The code in there is real code and must
  // keep its punctuation, unlike the surrounding text.
  let interpolation = 0;

  for (const line of lines) {
    let result = '';
    let index = 0;

    while (index < line.length) {
      const char = line[index];
      const next = line[index + 1];

      if (quote && interpolation === 0) {
        if (char === '\\') {
          // Keep the pair, blanked, so an escaped quote cannot end the string.
          result += '  ';
          index += 2;
          continue;
        }

        // Interpolation suspends the string: what follows is code again, and
        // its braces have to balance or every block after it drifts.
        if (quote === '`' && char === '$' && next === '{') {
          interpolation = 1;
          result += ' {';
          index += 2;
          continue;
        }

        if (char === quote) {
          quote = null;
          result += char;
        } else {
          result += ' ';
        }
        index++;
        continue;
      }

      // Inside `${ ... }`, braces are tracked so the matching one returns us
      // to the surrounding template text.
      if (interpolation > 0) {
        if (char === '{') interpolation++;
        if (char === '}') interpolation--;
        result += char;
        index++;
        continue;
      }

      if (QUOTES.has(char)) {
        quote = char;
        result += char;
        index++;
        continue;
      }

      result += char;
      index++;
    }

    // An unterminated ordinary string is a broken line, not a string that
    // continues; only a template literal really carries on to the next line.
    if (quote !== '`') {
      quote = null;
      interpolation = 0;
    }

    cleaned.push(result);
  }

  return cleaned.join('\n');
}

/**
 * Collapses formatting differences so indentation and spacing choices do not
 * make two identical blocks look different.
 */
export function normalize(source: string): string {
  const lines = source.split('\n');
  const kept: string[] = [];

  for (const line of lines) {
    const collapsed = line.replace(/\s+/g, ' ').trim();
    if (collapsed.length > 0) kept.push(collapsed);
  }

  return kept.join('\n');
}

/** How many lines carry actual code, ignoring blanks and comment-only lines. */
export function countSignificantLines(source: string): number {
  const normalized = normalize(stripComments(source));
  return normalized.length === 0 ? 0 : normalized.split('\n').length;
}

/**
 * The identity of a block of code, independent of how it is formatted or
 * commented. Equal hashes mean the same code appears twice.
 */
export function hashNormalized(source: string): string {
  const normalized = normalize(stripComments(source));
  return createHash('sha256').update(normalized).digest('hex').substring(0, 32);
}
