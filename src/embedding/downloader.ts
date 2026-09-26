import * as fs from 'fs';
import * as path from 'path';
import { MODEL_REPOSITORY, ModelStore, requiredFiles } from './model-store.js';
import type { ModelSettings } from '../config.js';

const HUGGING_FACE_BASE = 'https://huggingface.co';

/**
 * Fetches the embedding model so analysis can run.
 *
 * Kept a deliberate, separate step rather than something a question triggers:
 * a few hundred megabytes arriving unannounced in the middle of a developer's
 * work is worse than being told once that a command needs running.
 */
export async function downloadModel(
  settings: ModelSettings,
  report: (message: string) => void = () => {}
): Promise<void> {
  const store = new ModelStore(settings);
  const files = requiredFiles(settings.precision);

  report(`Downloading ${MODEL_REPOSITORY} (${settings.precision}) to ${store.modelDirectory}`);
  fs.mkdirSync(store.modelDirectory, { recursive: true });

  for (const file of files) {
    const target = path.join(store.modelDirectory, file);

    if (fs.existsSync(target) && fs.statSync(target).size > 0) {
      report(`  skipping ${file} (already present)`);
      continue;
    }

    await downloadFile(`${HUGGING_FACE_BASE}/${MODEL_REPOSITORY}/resolve/main/${file}`, target, file, report);
  }

  const status = store.status();
  if (status !== 'ready') {
    throw new Error(
      `Download finished but the model is still ${status}. Check the files under ` +
        `${store.modelDirectory} and try again.`
    );
  }

  report('Model ready.');
}

/**
 * Writes one file to disk, via a temporary name.
 *
 * The rename at the end is what makes an interrupted download safe: a partial
 * file never takes the real name, so it is never mistaken for a working model.
 */
async function downloadFile(
  url: string,
  target: string,
  label: string,
  report: (message: string) => void
): Promise<void> {
  const response = await fetch(url);

  if (!response.ok || !response.body) {
    throw new Error(`Could not download ${label}: HTTP ${response.status} ${response.statusText}`);
  }

  const expected = Number(response.headers.get('content-length') ?? 0);
  const temporary = `${target}.partial`;
  fs.mkdirSync(path.dirname(target), { recursive: true });

  const handle = fs.createWriteStream(temporary);
  let received = 0;
  let lastReported = 0;

  try {
    for await (const piece of response.body as unknown as AsyncIterable<Uint8Array>) {
      received += piece.length;
      await write(handle, piece);

      // Progress every few megabytes: often enough to show life, rarely enough
      // not to bury the terminal.
      if (received - lastReported > 8 * 1024 * 1024) {
        lastReported = received;
        report(`  ${label}: ${describeProgress(received, expected)}`);
      }
    }
  } finally {
    await new Promise<void>((resolve) => handle.end(resolve));
  }

  if (expected > 0 && received !== expected) {
    fs.rmSync(temporary, { force: true });
    throw new Error(
      `Download of ${label} was incomplete (${received} of ${expected} bytes). Try again.`
    );
  }

  fs.renameSync(temporary, target);
  report(`  ${label}: done (${formatBytes(received)})`);
}

function write(stream: fs.WriteStream, chunk: Uint8Array): Promise<void> {
  return new Promise((resolve, reject) => {
    stream.write(chunk, (error) => (error ? reject(error) : resolve()));
  });
}

function describeProgress(received: number, expected: number): string {
  if (expected <= 0) return formatBytes(received);
  const percent = Math.floor((received / expected) * 100);
  return `${formatBytes(received)} of ${formatBytes(expected)} (${percent}%)`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
