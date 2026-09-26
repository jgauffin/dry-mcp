import { defineConfig } from 'vitest/config';

/**
 * Integration tests need the embedding model on disk, so they are kept out of
 * the default run: CI stays offline and fast.
 */
export default defineConfig({
  test: {
    include: ['tests/integration/**/*.test.ts'],
    environment: 'node',
    testTimeout: 120000,
  },
});
