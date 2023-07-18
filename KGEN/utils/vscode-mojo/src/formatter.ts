//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import {exec} from 'child_process';
import * as vscode from 'vscode';

import {get} from './config';

export function registerFormatter(outputChannel: vscode.OutputChannel,
                                  extension: string) {
  return vscode.languages.registerDocumentFormattingEditProvider(extension, {
    provideDocumentFormattingEdits(document, _options) {
      const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
      const backupFolder = vscode.workspace.workspaceFolders?.[0];
      const cwd = workspaceFolder?.uri?.fsPath || backupFolder?.uri.fsPath;

      // Get the arguments to pass to the formatter.
      const args = get<string[]>('formatting.args', workspaceFolder, []);
      return new Promise<vscode.TextEdit[]>((resolve, reject) => {
        const originalDocumentText = document.getText();
        const command = "mojo format --quiet " + args.join(' ') + ' -';
        const process = exec(command, {cwd}, (error, stdout, stderr) => {
          // Process any errors/warnings during formatting. These aren't all
          // necessarily fatal, so this doesn't prevent edits from being
          // applied.
          if (error) {
            outputChannel.appendLine(`Formatting error:\n${stderr}`);
            reject(error);
          }

          // Formatter returned nothing, don't try to apply any edits.
          if (originalDocumentText.length > 0 && stdout.length === 0) {
            resolve([]);
            return;
          }

          // Otherwise, the formatter returned the formatted text. Update the
          // document.
          const documentRange = new vscode.Range(
              document.lineAt(0).range.start,
              document.lineAt(document.lineCount - 1)
                  .rangeIncludingLineBreak.end,
          );
          resolve([ new vscode.TextEdit(documentRange, stdout) ]);
        });

        process.stdin?.write(originalDocumentText);
        process.stdin?.end();
      });
    },
  });
}
