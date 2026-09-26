/**
 * What each tool accepts, as advertised to the calling agent.
 */
export const TOOL_SCHEMAS = {
  detectDuplication: {
    type: 'object',
    properties: {
      topN: {
        type: 'number',
        description: 'How many duplications to return. Default: 20.',
      },
      minLines: {
        type: 'number',
        description:
          'Ignore duplicated blocks shorter than this many lines. Default: 6 (or the ' +
          'project configuration). Raise it to see only the substantial findings.',
      },
      orderBy: {
        type: 'string',
        enum: ['severity', 'frequency'],
        description:
          'severity (default) ranks by how much the duplication costs, weighing block size ' +
          'above repetition. frequency ranks purely by how many copies exist.',
      },
      includeSuppressed: {
        type: 'boolean',
        description:
          'Also return the groups treated as accepted idioms, each with the reason it was ' +
          'demoted. Default: false. Use when tuning the idiom rules.',
      },
      minConfidence: {
        type: 'string',
        enum: ['certain', 'high', 'moderate', 'low'],
        description:
          'Leave out findings less certain than this. Default: "low", meaning everything ' +
          'is reported, because a large repeated block missed entirely is worse than one ' +
          'you have to check. Use "high" for only findings safe to act on without reading ' +
          'them first, or "certain" for exact copies alone.',
      },
      similarityThreshold: {
        type: 'number',
        minimum: 0,
        maximum: 1,
        description:
          'How alike two blocks must be to count as the same code, from 0 to 1. Default: ' +
          '0.45, measured against this model — code that is the same logic with every ' +
          'name changed scores about 0.53, the same logic in another language about 0.87, ' +
          'while unrelated code stays at or below 0.22. Lower it to surface more ' +
          'candidates at the cost of false positives; raise it toward 0.7 for near-copies ' +
          'only. Exact copies are found regardless of this setting.',
      },
    },
    additionalProperties: false,
  },

  duplicationStatus: {
    type: 'object',
    properties: {},
    additionalProperties: false,
  },

  explainDuplication: {
    type: 'object',
    properties: {
      clusterId: {
        type: 'string',
        description: 'The id of a duplication returned by detect_duplication.',
      },
    },
    required: ['clusterId'],
    additionalProperties: false,
  },

  reindex: {
    type: 'object',
    properties: {
      wait: {
        type: 'boolean',
        description:
          'Wait for the rebuild to finish before replying. Slow on a large project; without ' +
          'it the rebuild continues in the background.',
      },
    },
    additionalProperties: false,
  },
} as const;
