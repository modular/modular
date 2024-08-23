import * as assert from 'assert';

import * as vscode from 'vscode';

import { getExtension } from '../extension';
import path = require('path');
import { firstValueFrom } from 'rxjs';

const repoConfig = {
  root: path.join(__dirname, '..', '..', '..', '..', '..'),
  fixtures: path.join(__dirname, '..', '..', 'fixtures'),
};

function openModularRoot() {
  vscode.workspace.updateWorkspaceFolders(
    vscode.workspace.workspaceFolders?.length || 0,
    null,
    {
      uri: vscode.Uri.file(repoConfig.root),
      name: 'modular',
    }
  );
}

suite('LSP', () => {
  test('LSP should not be loaded on startup', async () => {
    const extension = await getExtension();
    assert.strictEqual(extension.mojoContext.lspContext!.lspClient, undefined);
  });

  test('LSP should be launched upon a file is opened', async () => {
    const extension = await getExtension();

    const lsp = firstValueFrom(
      extension.mojoContext.lspContext!.lspClientChanges
    );

    openModularRoot();
    await vscode.workspace.openTextDocument(
      vscode.Uri.file(
        path.join(repoConfig.fixtures, 'dangling-file', 'dangling-file.mojo')
      )
    );

    assert.strictEqual((await lsp)!.name, 'Mojo Language Client');
  });
});
