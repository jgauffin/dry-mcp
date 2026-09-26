import type { DuplicationService } from '../analysis/duplication-service.js';
import { duplicationTools, type ToolDefinition, type ToolHandlerFn } from './duplication-tools.js';
import { toYaml } from '../yaml.js';

export type { ToolDefinition, ToolHandlerFn, ToolModule } from './duplication-tools.js';

/** Every tool this server advertises. */
export const TOOL_DEFINITIONS: readonly ToolDefinition[] = duplicationTools.definitions;

const HANDLERS: Record<string, ToolHandlerFn> = duplicationTools.handlers;

/**
 * Routes MCP tool calls to the analysis, one at a time.
 *
 * Requests are serialized because they share one index and one cache; two
 * analyses running over a half-updated cache would disagree with each other for
 * no good reason.
 */
export class ToolHandler {
  private service: DuplicationService;
  private requestQueue: Promise<unknown> = Promise.resolve();
  private lastRefresh = 0;
  /** How long a scan of the tree stays good enough to reuse. */
  private static REFRESH_INTERVAL_MS = 2000;

  constructor(service: DuplicationService) {
    this.service = service;
  }

  handleTool(
    name: string,
    args: Record<string, unknown>
  ): Promise<{ content: Array<{ type: 'text'; text: string }>; isError?: boolean }> {
    const task = this.requestQueue.then(() => this.execute(name, args));
    this.requestQueue = task.catch(() => {});
    return task;
  }

  private async execute(
    name: string,
    args: Record<string, unknown>
  ): Promise<{ content: Array<{ type: 'text'; text: string }>; isError?: boolean }> {
    try {
      const handler = HANDLERS[name];
      if (!handler) throw new Error(`Unknown tool: ${name}`);

      // Notice edits before answering, but not on every call: walking the tree
      // repeatedly for a burst of questions costs more than it is worth.
      const now = Date.now();
      if (now - this.lastRefresh >= ToolHandler.REFRESH_INTERVAL_MS) {
        this.lastRefresh = now;
        this.service.refresh();
      }

      // Answering is the only time somebody is waiting, so it is the only time
      // the server is allowed to use the machine freely.
      const result = await this.service.whileAnswering(async () => handler(args, this.service));
      return { content: [{ type: 'text', text: toYaml(result) }] };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      // Marking the failure lets an agent tell a broken call from a finding of
      // "no duplication", which otherwise read the same.
      return {
        content: [{ type: 'text', text: `Error: ${message}` }],
        isError: true,
      };
    }
  }

  dispose(): void {
    this.service.dispose();
  }
}
