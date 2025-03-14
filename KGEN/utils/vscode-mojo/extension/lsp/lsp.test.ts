//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Note: extension tests disable the use of modular cli or magic SDK

import * as assert from 'assert';
import * as vscode from 'vscode';
import path = require('path');
import { firstValueFrom } from 'rxjs';
import { extension } from '../extension';

const repoConfig = {
  fixtures: path.join(__dirname, '..', '..', 'fixtures'),
};

suite('LSP', () => {
  test('LSP should not be loaded on startup', async () => {
    // Restart the extension. Tests run in a shared environment, so if other tests
    // have created the LSP, this test will fail otherwise.
    await vscode.commands.executeCommand('mojo.extension.restart');

    assert.strictEqual(extension.lspManager!.lspClient, undefined);
  });

  test('LSP should be launched when a Mojo file is opened', async () => {
    const lsp = firstValueFrom(extension.lspManager!.lspClientChanges);

    await vscode.workspace.openTextDocument(
      vscode.Uri.file(
        path.join(repoConfig.fixtures, 'dangling-file', 'dangling_file.mojo'),
      ),
    );

    assert.strictEqual((await lsp)!.name, 'Mojo Language Client');
  });
});
