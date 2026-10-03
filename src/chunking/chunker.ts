import type { Chunk } from '../types.js';
import {
  structuralView,
  hashNormalized,
  countSignificantLines,
  normalize,
} from './normalizer.js';

export interface ChunkerOptions {
  /** Blocks below this many significant lines are not worth reporting. */
  minLines: number;
  /**
   * The most lines to compare as a single unit. Longer blocks are cut into
   * overlapping windows so a big clone is still matched piece by piece.
   */
  maxLines: number;
}

/** How far consecutive windows of an oversized block overlap, as a fraction. */
const WINDOW_OVERLAP = 0.5;


interface OpenBlock {
  startLine: number;
  depth: number;
  indent: number;
  /** Whether indentation opened it, and so whether a dedent should close it. */
  byIndentation: boolean;
}

/** Nesting a line leaves open, and nesting it closes that began earlier. */
interface NetNesting {
  opened: number;
  closed: number;
}

/**
 * Splits source files into the blocks worth comparing for duplication.
 *
 * Works without a parser so every language the team writes is covered, at the
 * cost of inferring block boundaries rather than knowing them. Two signals
 * carry it: nesting punctuation for brace languages, and indentation for the
 * rest. Boundaries are therefore best-effort, and results say so.
 */
export class Chunker {
  private options: ChunkerOptions;

  constructor(options: ChunkerOptions) {
    this.options = options;
  }

  /**
   * Finds every candidate block in one file.
   *
   * Nested blocks are emitted alongside their parents on purpose: a duplicated
   * loop inside two otherwise different functions is exactly the kind of
   * duplication a developer wants told about.
   */
  chunk(file: string, source: string): Chunk[] {
    const lines = source.split('\n');
    const codeLines = structuralView(source).split('\n');
    const chunks: Chunk[] = [];
    const open: OpenBlock[] = [];

    let depth = 0;

    for (let index = 0; index < codeLines.length; index++) {
      const codeLine = codeLines[index];
      const trimmed = codeLine.trim();

      // Structure is measured in braces alone. Parentheses and brackets wrap
      // continuations of a single statement, not bodies, so letting them move
      // the depth would start blocks that never close.
      const net = this.netNesting(codeLine);

      if (net.closed > 0) {
        depth -= net.closed;
        // A block ends once nesting falls back below the depth inside it. The
        // line carrying the closer belongs to the block it closes.
        while (
          open.length > 0 &&
          !open[open.length - 1].byIndentation &&
          open[open.length - 1].depth > depth
        ) {
          const block = open.pop()!;
          this.emit(chunks, file, lines, block.startLine, index + 1);
        }
      }

      if (trimmed.length > 0) {
        const indent = this.indentOf(codeLine);

        // A line dedenting to or past an open block ends it, which is what
        // carries languages marking structure by indentation rather than
        // braces. Only blocks opened that way are closed here: a braced block
        // is ended by its closing brace, whatever the indentation does.
        while (
          open.length > 0 &&
          open[open.length - 1].byIndentation &&
          open[open.length - 1].depth === depth &&
          indent <= open[open.length - 1].indent &&
          open[open.length - 1].startLine < index
        ) {
          const block = open.pop()!;
          this.emit(chunks, file, lines, block.startLine, index);
        }

        if (net.opened > 0) {
          // Record the depth *inside* the block. The block ends when nesting
          // falls back below that, which is precisely its closing bracket.
          open.push({
            startLine: index + 1,
            depth: depth + net.opened,
            indent,
            byIndentation: false,
          });
        } else if (net.opened === 0 && this.opensByIndentation(codeLines, index)) {
          open.push({ startLine: index + 1, depth, indent, byIndentation: true });
        }
      }

      depth += net.opened;
    }

    // Whatever is still open runs to the end of the file.
    while (open.length > 0) {
      const block = open.pop()!;
      this.emit(chunks, file, lines, block.startLine, codeLines.length);
    }

    return this.deduplicate(chunks);
  }

  /**
   * Records a block, splitting it when it is too long to compare in one piece.
   */
  private emit(
    chunks: Chunk[],
    file: string,
    lines: string[],
    startLine: number,
    endLine: number
  ): void {
    const text = lines.slice(startLine - 1, endLine).join('\n');
    const significant = countSignificantLines(text);

    if (significant < this.options.minLines) return;

    if (significant <= this.options.maxLines) {
      chunks.push(this.build(file, lines, startLine, endLine));
      return;
    }

    for (const window of this.windows(startLine, endLine)) {
      const windowText = lines.slice(window.start - 1, window.end).join('\n');
      if (countSignificantLines(windowText) < this.options.minLines) continue;
      // Each window remembers the block it was cut from, so two windows of one
      // block are never mistaken for copies of each other.
      chunks.push({
        ...this.build(file, lines, window.start, window.end),
        blockStartLine: startLine,
        blockEndLine: endLine,
      });
    }
  }

  /**
   * Cuts an oversized block into overlapping windows.
   *
   * They overlap so a duplicated stretch straddling two window boundaries is
   * still wholly contained in some window, rather than split across two and
   * matched by neither.
   */
  private windows(startLine: number, endLine: number): { start: number; end: number }[] {
    const size = this.options.maxLines;
    const step = Math.max(1, Math.floor(size * (1 - WINDOW_OVERLAP)));
    const result: { start: number; end: number }[] = [];

    for (let start = startLine; start <= endLine; start += step) {
      const end = Math.min(start + size - 1, endLine);
      result.push({ start, end });
      if (end === endLine) break;
    }

    return result;
  }

  private build(file: string, lines: string[], startLine: number, endLine: number): Chunk {
    const text = lines.slice(startLine - 1, endLine).join('\n');
    return {
      file,
      startLine,
      endLine,
      text,
      normalizedHash: hashNormalized(text),
      significantLines: countSignificantLines(text),
    };
  }

  /**
   * Drops blocks covering exactly the same lines, which the brace and
   * indentation signals both report when they agree.
   */
  private deduplicate(chunks: Chunk[]): Chunk[] {
    const seen = new Set<string>();
    const unique: Chunk[] = [];

    for (const chunk of chunks) {
      const key = `${chunk.startLine}:${chunk.endLine}`;
      if (seen.has(key)) continue;
      seen.add(key);
      unique.push(chunk);
    }

    unique.sort((a, b) => a.startLine - b.startLine || a.endLine - b.endLine);
    return unique;
  }

  /**
   * How much brace nesting a line opens, and how much it closes that began
   * earlier. Braces that open and close on the same line cancel out, so a
   * one-line object literal is not mistaken for a block.
   */
  private netNesting(line: string): NetNesting {
    let opened = 0;
    let closed = 0;

    for (const char of line) {
      if (char === '{') {
        opened++;
      } else if (char === '}') {
        if (opened > 0) {
          opened--;
        } else {
          closed++;
        }
      }
    }

    return { opened, closed };
  }


  /**
   * Whether a line ends the way a block header does in an indentation-based
   * language: a colon for Python, or a dash or colon for YAML-like data.
   */
  private endsWithBodyIntroducer(line: string): boolean {
    const trimmed = line.trimEnd();
    return trimmed.endsWith(':') || trimmed.endsWith('-');
  }

  private indentOf(line: string): number {
    const match = line.match(/^[ \t]*/);
    if (!match) return 0;
    // A tab stands for one level, the same as a run of spaces would.
    return match[0].replace(/\t/g, '    ').length;
  }

  /**
   * Whether this line introduces an indented body, the way `def f():` does.
   *
   * Requires the line to end in a token that actually opens one. Judging by
   * indentation alone would treat any statement that happens to be followed by
   * something more indented as the start of a block — inside braced code that
   * is most of the file, and each one becomes a block running to the end of its
   * enclosing scope.
   */
  private opensByIndentation(codeLines: string[], index: number): boolean {
    const current = codeLines[index];
    if (current.trim().length === 0) return false;
    if (!this.endsWithBodyIntroducer(current)) return false;

    const currentIndent = this.indentOf(current);

    for (let next = index + 1; next < codeLines.length; next++) {
      const candidate = codeLines[next];
      if (candidate.trim().length === 0) continue;
      return this.indentOf(candidate) > currentIndent;
    }

    return false;
  }
}

/** Re-exported so callers hashing ad-hoc text share the chunker's definition. */
export { hashNormalized, normalize };
