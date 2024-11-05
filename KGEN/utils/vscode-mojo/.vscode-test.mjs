import { defineConfig } from '@vscode/test-cli';

const baseConfig = {
  platform: 'desktop',
  workspaceFolder: '../../../',
  version: '1.92.2',
  files: 'out/**/*.test.js',
  mocha: {
    timeout: 30000, // 30 seconds
    reporter: 'out/test/reporter.js',
  },
};

export default defineConfig([
  {
    ...baseConfig,
    label: 'dev',
  },
  {
    ...baseConfig,
    // CI environments don't have a monorepo.
    workspaceFolder: './',
    mocha: {
      // Longer timeout because this will download Magic + the SDK.
      timeout: 5 * 60 * 1000,
      reporter: 'out/test/reporter.js',
    },
    label: 'ci',
    env: {
      // Force magic download.
      MOJO_EXTENSION_FORCE_MAGIC: '1',
    },
  },
]);
