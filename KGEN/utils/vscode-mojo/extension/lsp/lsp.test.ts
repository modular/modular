//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Note: extension tests disable the use of modular cli or magic SDK

import * as assert from 'assert';
import * as vscode from 'vscode';
import { extension } from '../extension';
import path = require('path');
import { firstValueFrom } from 'rxjs';

const repoConfig = {
  root: path.join(__dirname, '..', '..', '..', '..', '..'),
  fixtures: path.join(__dirname, '..', '..', 'fixtures'),
};

function openModularRoot() {
  vscode.workspace.updateWorkspaceFolders(
    vscode.workspace.workspaceFolders?.length || 0,
    undefined,
    {
      uri: vscode.Uri.file(repoConfig.root),
      name: 'modular',
    }
  );
}

suite('LSP', () => {
  test('LSP should not be loaded on startup', async () => {
    assert.strictEqual(extension.lspManager!.lspClient, undefined);
  });

  test('LSP should be launched upon a file is opened', async () => {
    const lsp = firstValueFrom(extension.lspManager!.lspClientChanges);

    openModularRoot();
    await vscode.workspace.openTextDocument(
      vscode.Uri.file(
        path.join(repoConfig.fixtures, 'dangling-file', 'dangling_file.mojo')
      )
    );

    assert.strictEqual((await lsp)!.name, 'Mojo Language Client');
  });
});
