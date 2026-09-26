import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import type { ModelStatus } from '../types.js';
import type { ModelSettings } from '../config.js';

/** The repository the embeddings come from, in Hugging Face's naming. */
export const MODEL_REPOSITORY = 'jinaai/jina-embeddings-v2-base-code';

/** Weight files, one per precision the server offers. */
const WEIGHT_FILES: Record<ModelSettings['precision'], string> = {
  int8: 'onnx/model_quantized.onnx',
  fp32: 'onnx/model.onnx',
};

/**
 * Files the model cannot load without. Sizes guard against a download that was
 * interrupted: a truncated weight file otherwise fails deep inside the ONNX
 * runtime, where the error says nothing about what the developer should do.
 */
const REQUIRED_SUPPORT_FILES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
];

/** Below this, a weight file is certainly a partial download. */
const MINIMUM_WEIGHT_BYTES = 50 * 1024 * 1024;

export interface ModelLocation {
  /** The root Transformers.js resolves repository names against. */
  rootDirectory: string;
  /** Where this specific model's files live. */
  modelDirectory: string;
  precision: ModelSettings['precision'];
}

/**
 * Knows whether the embedding model is available, and says so plainly when it
 * is not.
 *
 * The model is a few hundred megabytes, so it is downloaded deliberately rather
 * than pulled in behind the developer's back the first time they ask a
 * question. That makes "not installed" a normal state the server has to answer
 * from, not an error.
 */
export class ModelStore {
  private location: ModelLocation;

  constructor(settings: ModelSettings) {
    this.location = resolveLocation(settings);
  }

  get rootDirectory(): string {
    return this.location.rootDirectory;
  }

  get modelDirectory(): string {
    return this.location.modelDirectory;
  }

  get precision(): ModelSettings['precision'] {
    return this.location.precision;
  }

  /**
   * Identifies which model produced a vector, so a precision or model change
   * invalidates the cache instead of silently mixing incomparable vectors.
   */
  get modelId(): string {
    return `${MODEL_REPOSITORY}@${this.location.precision}`;
  }

  /** The file name Transformers.js should load for the chosen precision. */
  get weightFileName(): string {
    return WEIGHT_FILES[this.location.precision];
  }

  /**
   * Whether the model can be loaded: fully present, partly downloaded, or
   * absent altogether.
   */
  status(): ModelStatus {
    const weightPath = path.join(this.location.modelDirectory, this.weightFileName);

    if (!fs.existsSync(weightPath)) {
      // Support files without weights still counts as an interrupted download,
      // which is worth distinguishing from having never started.
      const anySupport = REQUIRED_SUPPORT_FILES.some((file) =>
        fs.existsSync(path.join(this.location.modelDirectory, file))
      );
      return anySupport ? 'incomplete' : 'not-installed';
    }

    for (const file of REQUIRED_SUPPORT_FILES) {
      if (!fs.existsSync(path.join(this.location.modelDirectory, file))) return 'incomplete';
    }

    try {
      if (fs.statSync(weightPath).size < MINIMUM_WEIGHT_BYTES) return 'incomplete';
    } catch {
      return 'incomplete';
    }

    return 'ready';
  }

  /**
   * What to tell the developer when analysis cannot run, phrased as the action
   * that fixes it.
   */
  explainUnavailable(): string {
    const size = this.location.precision === 'int8' ? '~160 MB' : '~640 MB';
    const flag = this.location.precision === 'fp32' ? ' --fp32' : '';

    // Both forms, because the bare command only exists once the package is
    // installed globally, and some environments block npx outright.
    const command = `duplication-mcp download-model${flag}`;

    if (this.status() === 'incomplete') {
      return (
        `The embedding model at ${this.location.modelDirectory} is incomplete, which ` +
        `usually means an interrupted download. Run '${command}' (or ` +
        `'node dist/index.js download-model${flag}' from the install directory), ` +
        `${size}, to finish it.`
      );
    }

    return (
      `Duplication analysis needs the ${MODEL_REPOSITORY} model, which is not installed. ` +
      `Run '${command}' (or 'node dist/index.js download-model${flag}' from the install ` +
      `directory), ${size}, to enable it.`
    );
  }
}

/**
 * Where the model should be found.
 *
 * Transformers.js resolves a repository name against a root directory, so the
 * layout on disk has to mirror the `owner/name` form the repository uses.
 */
export function resolveLocation(settings: ModelSettings): ModelLocation {
  const rootDirectory =
    settings.path ??
    process.env.DUPLICATION_MODEL_PATH ??
    path.join(os.homedir(), '.cache', 'duplication-mcp', 'models');

  return {
    rootDirectory,
    modelDirectory: path.join(rootDirectory, ...MODEL_REPOSITORY.split('/')),
    precision: settings.precision,
  };
}

/** Every file a working installation needs, for the downloader to fetch. */
export function requiredFiles(precision: ModelSettings['precision']): string[] {
  return [...REQUIRED_SUPPORT_FILES, WEIGHT_FILES[precision]];
}
