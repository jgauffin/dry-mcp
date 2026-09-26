#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { startServer } from './server.js';
import { downloadModel } from './embedding/downloader.js';
import { loadConfig } from './config.js';
import { ModelStore } from './embedding/model-store.js';
import { resolveProxy, proxyEnvironment } from './embedding/network.js';

/**
 * Entry point. Runs the MCP server, or the one-off commands a developer needs
 * before the server can do anything useful.
 */
async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const command = args[0];

  if (command === 'download-model') {
    await runDownload(args);
    return;
  }

  if (command === '--help' || command === '-h') {
    printUsage();
    return;
  }

  const projectRoot = command && !command.startsWith('-') ? command : process.cwd();
  await startServer(projectRoot);
}

/** Fetches the embedding model, reporting progress on stderr. */
async function runDownload(args: string[]): Promise<void> {
  const projectRoot = process.cwd();
  const config = loadConfig(projectRoot);
  const precision = args.includes('--fp32') ? 'fp32' : config.model.precision;
  const settings = { ...config.model, precision: precision as 'int8' | 'fp32' };

  const store = new ModelStore(settings);
  if (store.status() === 'ready') {
    console.error(`Model already installed at ${store.modelDirectory}`);
    return;
  }

  await withProxy(args, () => downloadModel(settings, (message) => console.error(message)));
}

/**
 * Runs the download with this machine's proxy in force.
 *
 * On a corporate network the way out is usually a proxy the developer has never
 * had to configure by hand. Node only reads the proxy environment at startup,
 * so when one is discovered rather than already set, the command restarts
 * itself with that environment instead of failing on a bare connection reset.
 */
async function withProxy(args: string[], download: () => Promise<void>): Promise<void> {
  const alreadyConfigured = process.env.NODE_USE_ENV_PROXY === '1';
  if (alreadyConfigured) {
    await download();
    return;
  }

  const proxy = resolveProxy();
  if (!proxy) {
    await download();
    return;
  }

  console.error(`Using proxy ${proxy}`);
  const result = spawnSync(process.execPath, [process.argv[1], ...args], {
    env: proxyEnvironment(proxy),
    stdio: 'inherit',
  });

  if (result.status !== 0) {
    throw new Error('Download failed. See the messages above.');
  }
}

function printUsage(): void {
  console.error(
    [
      'duplication-mcp — finds duplicated code, ranked by what it costs to keep',
      '',
      'Usage:',
      '  duplication-mcp [project-root]        Run the MCP server over a project',
      '  duplication-mcp download-model        Download the embedding model (~160 MB)',
      '  duplication-mcp download-model --fp32 Download full precision instead (~640 MB)',
      '',
      'The server runs without the model, but reports no duplications until it is',
      'installed.',
    ].join('\n')
  );
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Error: ${message}`);
  process.exit(1);
});
