import { defineConfig } from '@vscode/test-cli';

export default defineConfig({
  platform: 'desktop',
  workspaceFolder: '../../../',
  version: '1.92.2',
  files: 'out/**/*.test.js',
  mocha: {
    timeout: 5000, // 5 seconds
    reporter: 'out/test/reporter.js',
  },
});
