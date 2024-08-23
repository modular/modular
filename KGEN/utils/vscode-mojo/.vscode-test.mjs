import { defineConfig } from '@vscode/test-cli';

export default defineConfig({
  files: 'out/**/*.test.js',
  mocha: {
    timeout: 5000, // 5 seconds
  },
});
