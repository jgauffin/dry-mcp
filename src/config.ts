import * as fs from 'fs';
import * as path from 'path';

/**
 * Rules for deciding that repeated code is an accepted idiom rather than a
 * duplication worth fixing.
 */
export interface IdiomSettings {
  /** At or below this many lines, repetition is cheap to tolerate. */
  maxLines: number;
  /** Appearing at least this often suggests a language or framework habit. */
  minOccurrences: number;
  /** Content patterns the team has decided never to be told about again. */
  ignorePatterns: string[];
}

export interface ModelSettings {
  /**
   * int8 is the only precision CPUs genuinely accelerate; fp32 costs four times
   * the download for a small accuracy gain.
   */
  precision: 'int8' | 'fp32';
  /** Where the downloaded model lives, when not in the default cache. */
  path: string | null;
}

export interface DuplicationConfig {
  include: string[];
  exclude: string[];
  /** Blocks shorter than this are never worth reporting. */
  minLines: number;
  /** How alike two blocks must be to count as the same code. */
  similarityThreshold: number;
  idiom: IdiomSettings;
  model: ModelSettings;
}

export const DEFAULT_CONFIG: DuplicationConfig = {
  include: ['**'],
  exclude: [
    '**/node_modules/**',
    '**/dist/**',
    '**/build/**',
    '**/coverage/**',
    '**/*.min.*',
    '**/*.generated.*',
    '**/migrations/**',
  ],
  minLines: 6,
  /**
   * Measured against the model rather than guessed. Code that is the same
   * logic with every name changed scores about 0.53, the same logic written in
   * another language about 0.87, while genuinely different code — including a
   * loop of the same shape doing unrelated work — stays at or below 0.22.
   *
   * 0.45 sits inside that gap, nearer the negatives, so a rename is still
   * caught while a passing structural resemblance is not. Raise it to be told
   * about fewer, more certain findings.
   */
  similarityThreshold: 0.45,
  idiom: {
    maxLines: 5,
    minOccurrences: 10,
    ignorePatterns: [],
  },
  model: {
    precision: 'int8',
    path: null,
  },
};

const CONFIG_FILE_NAME = 'duplication.config.json';

/**
 * Reads the project's settings, falling back to defaults for anything the team
 * has not expressed an opinion about.
 *
 * A malformed config is reported rather than silently ignored: running with
 * defaults the developer did not choose would quietly change what gets flagged.
 */
export function loadConfig(projectRoot: string): DuplicationConfig {
  return readConfigFile(projectRoot) ?? DEFAULT_CONFIG;
}

/**
 * The project's settings as written down, or null when the team has not
 * written any.
 *
 * The difference matters to a caller re-reading the file to pick up edits: an
 * absent file means "whatever you were told to use", not "go back to the
 * defaults", so a caller given settings directly does not lose them.
 */
export function readConfigFile(projectRoot: string): DuplicationConfig | null {
  const configPath = path.join(projectRoot, CONFIG_FILE_NAME);

  let raw: string;
  try {
    raw = fs.readFileSync(configPath, 'utf-8');
  } catch {
    return null;
  }

  let parsed: Partial<DuplicationConfig>;
  try {
    parsed = JSON.parse(raw) as Partial<DuplicationConfig>;
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`${CONFIG_FILE_NAME} is not valid JSON: ${detail}`);
  }

  return mergeWithDefaults(parsed);
}

/**
 * Layers user settings over the defaults, one level deep for the nested groups
 * so naming a single idiom rule does not discard the others.
 */
export function mergeWithDefaults(parsed: Partial<DuplicationConfig>): DuplicationConfig {
  return {
    include: parsed.include ?? DEFAULT_CONFIG.include,
    exclude: parsed.exclude ?? DEFAULT_CONFIG.exclude,
    minLines: parsed.minLines ?? DEFAULT_CONFIG.minLines,
    similarityThreshold: parsed.similarityThreshold ?? DEFAULT_CONFIG.similarityThreshold,
    idiom: { ...DEFAULT_CONFIG.idiom, ...(parsed.idiom ?? {}) },
    model: { ...DEFAULT_CONFIG.model, ...(parsed.model ?? {}) },
  };
}
