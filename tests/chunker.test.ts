import { describe, it, expect } from 'vitest';
import { Chunker } from '../src/chunking/chunker.js';
import { stripComments, normalize } from '../src/chunking/normalizer.js';

describe('Chunker', () => {
  const chunker = new Chunker({ minLines: 3, maxLines: 400 });

  it('Braced_function_body_is_extracted_as_one_block', () => {
    const source = [
      'function calculateTotal(order) {',
      '  let total = 0;',
      '  for (const line of order.lines) {',
      '    total += line.price;',
      '  }',
      '  return total;',
      '}',
    ].join('\n');

    const chunks = chunker.chunk('src/order.js', source);

    const whole = chunks.find((c) => c.startLine === 1 && c.endLine === 7);
    expect(whole).toBeDefined();
  });

  it('Indentation_based_block_is_extracted_without_braces', () => {
    const source = [
      'def calculate_total(order):',
      '    total = 0',
      '    for line in order.lines:',
      '        total += line.price',
      '    return total',
    ].join('\n');

    const chunks = chunker.chunk('src/order.py', source);

    expect(chunks.length).toBeGreaterThan(0);
    const whole = chunks.find((c) => c.startLine === 1 && c.endLine === 5);
    expect(whole).toBeDefined();
  });

  it('Blocks_shorter_than_the_minimum_are_not_reported', () => {
    const source = ['function noop() {', '}'].join('\n');

    const chunks = chunker.chunk('src/tiny.js', source);

    expect(chunks).toHaveLength(0);
  });

  it('Line_numbers_still_point_at_the_original_source_after_comments_are_removed', () => {
    const source = [
      '// A leading comment that is not code.',
      '// Another one.',
      'function withComments() {',
      '  const a = 1; // trailing',
      '  const b = 2;',
      '  const c = 3;',
      '  return a + b + c;',
      '}',
    ].join('\n');

    const chunks = chunker.chunk('src/commented.js', source);

    const whole = chunks.find((c) => c.startLine === 3);
    expect(whole).toBeDefined();
    expect(whole!.endLine).toBe(8);
  });

  it('Identical_code_formatted_differently_produces_the_same_hash', () => {
    const compact = ['function f() {', '  const a = 1;', '  const b = 2;', '  return a + b;', '}'].join('\n');
    const spaced = [
      'function f() {',
      '',
      '      const a = 1;',
      '',
      '      const b = 2;',
      '      return a + b;',
      '}',
    ].join('\n');

    const first = chunker.chunk('src/a.js', compact)[0];
    const second = chunker.chunk('src/b.js', spaced)[0];

    expect(first.normalizedHash).toBe(second.normalizedHash);
  });

  it('Code_differing_only_in_comments_produces_the_same_hash', () => {
    const documented = [
      'function f() {',
      '  // Explains the first step.',
      '  const a = 1;',
      '  const b = 2;',
      '  return a + b;',
      '}',
    ].join('\n');
    const bare = ['function f() {', '  const a = 1;', '  const b = 2;', '  return a + b;', '}'].join('\n');

    const first = chunker.chunk('src/a.js', documented)[0];
    const second = chunker.chunk('src/b.js', bare)[0];

    expect(first.normalizedHash).toBe(second.normalizedHash);
  });

  it('Nested_blocks_are_reported_alongside_their_parent', () => {
    const source = [
      'function outer() {',
      '  const setup = 1;',
      '  if (setup) {',
      '    const a = 1;',
      '    const b = 2;',
      '    doSomething(a, b);',
      '  }',
      '  return setup;',
      '}',
    ].join('\n');

    const chunks = chunker.chunk('src/nested.js', source);

    expect(chunks.some((c) => c.startLine === 1)).toBe(true);
    expect(chunks.some((c) => c.startLine === 3)).toBe(true);
  });

  it('Oversized_block_is_windowed_so_a_large_clone_is_still_comparable', () => {
    const body: string[] = ['function huge() {'];
    for (let i = 0; i < 300; i++) {
      body.push(`  step${i}();`);
    }
    body.push('}');

    const windowing = new Chunker({ minLines: 3, maxLines: 50 });
    const chunks = windowing.chunk('src/huge.js', body.join('\n'));

    expect(chunks.length).toBeGreaterThan(1);
    for (const chunk of chunks) {
      expect(chunk.significantLines).toBeLessThanOrEqual(50);
    }
  });

  it('Each_window_remembers_the_block_it_was_cut_from', () => {
    const body: string[] = ['function huge() {'];
    for (let i = 0; i < 300; i++) {
      body.push(`  step${i}();`);
    }
    body.push('}');

    const windowing = new Chunker({ minLines: 3, maxLines: 50 });
    const chunks = windowing.chunk('src/huge.js', body.join('\n'));

    for (const chunk of chunks) {
      expect(chunk.blockStartLine).toBe(1);
      expect(chunk.blockEndLine).toBe(302);
    }
  });

  it('A_block_short_enough_to_compare_whole_is_its_own_block', () => {
    const source = ['function f() {', '  a();', '  b();', '  c();', '}'].join('\n');

    const [chunk] = chunker.chunk('src/f.js', source);

    expect(chunk.blockStartLine).toBeUndefined();
    expect(chunk.blockEndLine).toBeUndefined();
  });
});

describe('Comment stripping', () => {
  it('Line_comment_markers_inside_strings_are_kept_as_code', () => {
    const source = 'const url = "https://example.com/path"; // real comment';

    const stripped = stripComments(source);

    expect(stripped).toContain('https://example.com/path');
    expect(stripped).not.toContain('real comment');
  });

  it('Hash_comments_are_removed_for_shell_and_python_style_sources', () => {
    const source = 'value = 1  # explain';

    expect(stripComments(source)).not.toContain('explain');
  });

  it('Block_comments_spanning_lines_are_removed_but_line_count_is_preserved', () => {
    const source = ['const a = 1;', '/* first', '   second */', 'const b = 2;'].join('\n');

    const stripped = stripComments(source);

    expect(stripped.split('\n')).toHaveLength(4);
    expect(stripped).not.toContain('first');
    expect(stripped).not.toContain('second');
    expect(stripped).toContain('const b = 2;');
  });

  it('Sql_style_double_dash_comments_are_removed', () => {
    const source = 'SELECT 1 -- explain';

    expect(stripComments(source)).not.toContain('explain');
  });

  it('Decrement_operator_is_not_mistaken_for_a_sql_comment', () => {
    const source = 'count--;';

    expect(normalize(stripComments(source))).toContain('count--;');
  });
});
