//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import {exec} from 'child_process';
import * as vscode from 'vscode';

export function registerFormatter(outputChannel: vscode.OutputChannel,
                                  extension: string) {
  return vscode.languages.registerDocumentFormattingEditProvider(extension, {
    provideDocumentFormattingEdits(document, _options) {
      const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
      const backupFolder = vscode.workspace.workspaceFolders?.[0];
      const cwd = workspaceFolder?.uri?.fsPath || backupFolder?.uri.fsPath;

      // Get the arguments passed to black when formatting python code. We'll
      // try to use the same settings.
      const blackArgs =
          vscode.workspace
              .getConfiguration('python.formatting', workspaceFolder)
              .get<string[]>('blackArgs', []);

      return new Promise<vscode.TextEdit[]>((resolve, reject) => {
        const originalDocumentText = document.getText();
        const command =
            "mblack --fast --quiet " + blackArgs.join(' ') + ' -t mojo -';
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
