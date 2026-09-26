import type { DuplicationService, DuplicationQuery } from '../analysis/duplication-service.js';
import type { Confidence } from '../types.js';
import { TOOL_SCHEMAS } from './schemas.js';

/** An MCP tool as advertised to clients. */
export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: unknown;
}

export type ToolHandlerFn = (
  args: Record<string, unknown>,
  service: DuplicationService
) => unknown | Promise<unknown>;

export interface ToolModule {
  definitions: readonly ToolDefinition[];
  handlers: Record<string, ToolHandlerFn>;
}

/**
 * The tools this server offers, all of them about finding and understanding
 * repeated code.
 */
export const duplicationTools: ToolModule = {
  definitions: [
    {
      name: 'detect_duplication',
      description:
        'Find duplicated code, ranked by what it costs to keep. A large block copied a few ' +
        'times ranks above a small one repeated everywhere, since the latter is usually a ' +
        'language idiom. Matches both identical copies and near-misses such as renamed ' +
        'variables, using code embeddings.\n\n' +
        'Every finding carries a "confidence" of certain, high, moderate or low, and the ' +
        'reply includes a "confidenceGuide" explaining each. Near-miss matching is ' +
        'deliberately inclusive — by default (minConfidence "low", similarityThreshold ' +
        '0.45) everything is returned, because missing a large repeated block is worse ' +
        'than offering one that turns out to be coincidence. So expect some false ' +
        'positives at low and moderate, and read the code before acting on those; findings ' +
        'marked certain or high rarely need checking. Narrow with minConfidence or raise ' +
        'similarityThreshold when you want fewer, safer results.\n\n' +
        'Never waits for indexing. Embedding a large project takes minutes, so the reply ' +
        'carries a "progress" block (files in scope, files embedded, percent complete) ' +
        'whenever the index is incomplete; until half a large project is embedded it ' +
        'reports progress alone rather than a ranking that would mostly reflect which ' +
        'files were read first. Ask again shortly, or watch duplication_status.\n\n' +
        'When the scope looks wider than intended the reply also carries a "scope" block ' +
        'and says how to narrow it by editing duplication.config.json in the project root.\n\n' +
        'Block boundaries are inferred without a parser, so treat line ranges as ' +
        'approximate and read the returned source for the real extent.',
      inputSchema: TOOL_SCHEMAS.detectDuplication,
    },
    {
      name: 'duplication_status',
      description:
        'Report how ready the analysis is: indexing progress, blocks cached, whether the ' +
        'embedding model is installed, and what is in scope — the file count, the largest ' +
        'folders, and any nested repositories left out. Cheap — use it to follow indexing, ' +
        'or to see what would be analysed before narrowing the scope.',
      inputSchema: TOOL_SCHEMAS.duplicationStatus,
    },
    {
      name: 'explain_duplication',
      description:
        'Return the full source of every copy in one duplication, so it can be acted on. ' +
        'Takes the cluster id from detect_duplication.',
      inputSchema: TOOL_SCHEMAS.explainDuplication,
    },
    {
      name: 'reindex',
      description:
        'Re-embed the whole project from scratch. Use after changing the model or when the ' +
        'index is suspected stale; ordinary edits are picked up automatically.',
      inputSchema: TOOL_SCHEMAS.reindex,
    },
  ],

  handlers: {
    detect_duplication: (args, service) => {
      const query: DuplicationQuery = {
        topN: numberOr(args.topN, undefined),
        minLines: numberOr(args.minLines, undefined),
        orderBy: args.orderBy === 'frequency' ? 'frequency' : 'severity',
        includeSuppressed: args.includeSuppressed === true,
        minConfidence: confidenceOr(args.minConfidence),
        similarityThreshold: numberOr(args.similarityThreshold, undefined),
      };

      const report = service.analyze(query);
      // Keep working on the queue afterwards, so repeated questions converge on
      // a complete answer without the developer having to ask for a rebuild.
      service.startEmbedding();
      return report;
    },

    duplication_status: (_args, service) => {
      const status = service.status();
      const progress = service.progress(status);
      return {
        ...status,
        progress,
        scope: service.scopeSummary(progress.filesInScope),
        ready: status.modelStatus === 'ready' && status.pendingFiles === 0,
        analysisMode: 'heuristic',
      };
    },

    explain_duplication: (args, service) => {
      const clusterId = typeof args.clusterId === 'string' ? args.clusterId : '';
      const explanation = service.explain(clusterId);

      if (!explanation) {
        return {
          notice:
            `No duplication with id "${clusterId}" was found. An id stops resolving once ` +
            `the code it points at has been edited, so run detect_duplication again for a ` +
            `current one.`,
        };
      }

      return explanation;
    },

    reindex: async (args, service) => {
      service.reindex();

      if (args.wait === true) {
        await service.waitForEmbedding();
      } else {
        service.startEmbedding();
      }

      const status = service.status();
      const progress = service.progress(status);
      return {
        status,
        progress,
        notice:
          status.pendingFiles > 0
            ? `Rebuilding: ${progress.percentComplete}% done, ${status.pendingFiles} file(s) ` +
              `queued for embedding. It continues in the background; follow it with ` +
              `duplication_status.`
            : 'Rebuild complete.',
      };
    },
  },
};

function numberOr(value: unknown, fallback: number | undefined): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

const CONFIDENCE_LEVELS: Confidence[] = ['certain', 'high', 'moderate', 'low'];

function confidenceOr(value: unknown): Confidence | undefined {
  return CONFIDENCE_LEVELS.includes(value as Confidence) ? (value as Confidence) : undefined;
}
