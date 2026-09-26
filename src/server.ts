import * as path from 'path';
import * as os from 'os';
import { createHash } from 'crypto';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { ToolHandler, TOOL_DEFINITIONS } from './tools/index.js';
import { DuplicationService } from './analysis/duplication-service.js';
import { EmbeddingCache } from './cache/embedding-cache.js';
import { ModelStore } from './embedding/model-store.js';
import { CONFIG_FILE_NAME, loadConfig } from './config.js';
import { normalizePath } from './paths.js';
import { GitState } from './tools/git-state.js';

export interface ServerOptions {
  name?: string;
  description?: string;
}

const VERSION = '0.1.0';

/**
 * Builds the MCP server for a project.
 *
 * Starting up must be cheap and must never fail because the model is missing:
 * the developer can still ask for status and be told what to install.
 */
export function createServer(projectRoot?: string, options?: ServerOptions): McpServer {
  // Resolved because a caller may pass a relative path such as ".", and both
  // the cache key and the file walk need a real location.
  const resolvedRoot = path.resolve(projectRoot ?? process.cwd());
  const normalizedRoot = normalizePath(resolvedRoot);

  const config = loadConfig(resolvedRoot);
  const modelStore = new ModelStore(config.model);
  const cache = new EmbeddingCache(cachePathFor(normalizedRoot), modelStore.modelId);
  const service = new DuplicationService(resolvedRoot, config, cache, modelStore);
  const toolHandler = new ToolHandler(service);

  const mcpServer = new McpServer(
    {
      name: options?.name ?? 'duplication-mcp',
      version: VERSION,
      ...(options?.description ? { description: options.description } : {}),
    },
    {
      capabilities: { tools: {} },
      instructions: instructionsFor(normalizedRoot, modelStore),
    }
  );

  // A branch switch replaces the code under the index, so the answers it would
  // give describe work the developer has moved away from.
  const gitState = new GitState(resolvedRoot);
  const stopWatching = gitState.watch(() => {
    service.refresh();
    service.startEmbedding();
  });

  mcpServer.server.onclose = () => {
    stopWatching();
    toolHandler.dispose();
  };

  mcpServer.server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: TOOL_DEFINITIONS.map((tool) => ({
      name: tool.name,
      description: tool.description,
      inputSchema: tool.inputSchema,
    })),
  }));

  mcpServer.server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    try {
      return await toolHandler.handleTool(name, args ?? {});
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return {
        content: [{ type: 'text' as const, text: `Error: ${message}` }],
        isError: true,
      };
    }
  });

  // Start indexing straight away so the first question has something to answer
  // from, rather than beginning the work only when asked.
  service.refresh();
  service.startEmbedding();

  return mcpServer;
}

/**
 * Asks the operating system to schedule this server behind whatever the
 * developer is doing.
 *
 * Nothing here is ever urgent: the work is triggered by code changing, not by
 * anyone asking, so a compile or an editor should always win the core. This only
 * decides who waits when the machine is full, so it costs nothing when it is not.
 */
function yieldToTheDeveloper(): void {
  try {
    os.setPriority(os.constants.priority.PRIORITY_BELOW_NORMAL);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    console.error(`Could not lower process priority, continuing at normal: ${detail}`);
  }
}

/** What the calling agent needs to know before using these tools. */
function instructionsFor(root: string, modelStore: ModelStore): string {
  const lines = [
    `This server finds duplicated code in the project at: ${root}`,
    `Duplication is ranked by cost: a large block copied a few times ranks above a small`,
    `one repeated everywhere, because the latter is usually a language idiom rather than`,
    `a defect. Groups judged to be idioms are demoted and reported separately.`,
    ``,
    `Matching is language-agnostic and works without a parser, so block boundaries are`,
    `inferred rather than exact — treat reported line ranges as approximate and read the`,
    `returned source to establish the real extent.`,
    ``,
    `All file paths are relative to the project root, using forward slashes.`,
    ``,
    `Embedding happens in the background and is never waited for. An incomplete index is`,
    `reported as "progress" (files in scope, files embedded, percent complete); until half`,
    `of a large project is embedded, detect_duplication reports progress instead of`,
    `findings. Ask again shortly, or follow duplication_status.`,
    ``,
    `Scope is yours to set, in ${CONFIG_FILE_NAME} in the project root. Without that file`,
    `every source file under the root is analysed, which on a first run is usually more`,
    `than intended. Edit or create it directly — it is plain JSON and edits are picked up`,
    `before the next question, no restart:`,
    ``,
    `  { "include": ["src/**"], "exclude": ["vendor/**", "**/*.generated.*"] }`,
    ``,
    `Globs match project-relative paths with forward slashes, and exclude wins over`,
    `include. Nested repositories — git submodules and vendored clones — are left out by`,
    `default as other projects' code; "includeNestedRepositories": true brings them in.`,
  ];

  if (modelStore.status() !== 'ready') {
    lines.push('', `NOTE: ${modelStore.explainUnavailable()}`);
  }

  return lines.join('\n');
}

/**
 * Where a project's cache lives.
 *
 * Keyed by path so several projects can be analysed without their vectors
 * colliding, and kept outside the project so analysing a repository never
 * dirties the working tree.
 */
export function cachePathFor(projectRoot: string): string {
  const key = createHash('sha256').update(projectRoot.toLowerCase()).digest('hex').substring(0, 16);
  return path.join(os.homedir(), '.cache', 'duplication-mcp', `${key}.db`);
}

/** Starts the server on stdio, which is how MCP clients launch it. */
export async function startServer(projectRoot?: string, options?: ServerOptions): Promise<void> {
  // Clients commonly pass "." for the project they launched in. Resolving it
  // keeps the cache keyed by a real location — two projects both named "."
  // would otherwise share one index — and makes the startup line say where the
  // server is actually looking.
  const resolvedRoot = path.resolve(projectRoot ?? process.cwd());
  yieldToTheDeveloper();
  const mcpServer = createServer(resolvedRoot, options);

  await mcpServer.connect(new StdioServerTransport());

  // stdout carries the protocol, so progress goes to stderr.
  console.error(`duplication-mcp started for: ${resolvedRoot}`);
}
