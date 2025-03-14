//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as assert from 'assert';
import * as vscode from 'vscode';
import path = require('path');
import { firstValueFrom } from 'rxjs';
import { extension } from '../extension';

const repoConfig = {
  fixtures: path.join(__dirname, '..', '..', 'fixtures'),
};

suite('LSP recording', () => {
  test('LSP recording should generate a file in the workspace folder', async () => {
    const eventualLsp = firstValueFrom(extension.lspManager!.lspClientChanges);
    const fixtureUri = vscode.Uri.file(
      path.join(repoConfig.fixtures, 'dangling-file', 'dangling_file.mojo'),
    );

    await vscode.workspace.openTextDocument(fixtureUri);

    const lsp = await eventualLsp;
    assert.strictEqual(lsp!.name, 'Mojo Language Client');

    await vscode.commands.executeCommand('mojo.lsp.startRecord');

    const workspacePath = vscode.workspace.workspaceFolders![0]!.uri;
    const tracePath = vscode.Uri.joinPath(
      workspacePath,
      'mojo-lsp-recording.jsonl',
    );

    await lsp!.sendRequest('textDocument/hover', {
      textDocument: {
        uri: fixtureUri.toString(),
      },
      position: {
        line: 0,
        character: 0,
      },
    });

    const trace = (await vscode.workspace.fs.readFile(tracePath)).toString();
    const lines = trace.split('\n');
    assert.ok(lines.length >= 1, 'should record at least one request');
    const firstEntry = JSON.parse(lines[0]);
    assert.deepStrictEqual(firstEntry, {
      method: 'textDocument/hover',
      type: 'request',
      param: {
        textDocument: {
          uri: fixtureUri.toString(),
        },
        position: {
          line: 0,
          character: 0,
        },
      },
    });
  });
});
