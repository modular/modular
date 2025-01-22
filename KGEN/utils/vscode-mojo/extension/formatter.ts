//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import { execFile } from 'child_process';
import * as vscode from 'vscode';

import { MAXSDKManager } from './sdk/sdkManager';
import { get } from './utils/config';

export function registerFormatter(maxSDKManager: MAXSDKManager) {
  return vscode.languages.registerDocumentFormattingEditProvider('mojo', {
    async provideDocumentFormattingEdits(document, _options) {
      const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
      const backupFolder = vscode.workspace.workspaceFolders?.[0];
      const cwd = workspaceFolder?.uri?.fsPath || backupFolder?.uri.fsPath;
      const args = get<string[]>('formatting.args', workspaceFolder, []);

      // We use 'hideRepeatedErrors' because this action is often automated.
      const sdk = await maxSDKManager.findSDK(/*hideRepeatedErrors=*/ true);

      if (!sdk) {
        return [];
      }

      let env = sdk.getProcessEnv();

      return new Promise<vscode.TextEdit[]>(function (resolve, reject) {
        const originalDocumentText = document.getText();
        const process = execFile(
          sdk.config.mojoMBlackPath,
          ['--fast', '--preview', '--quiet', '-t', 'mojo', ...args, '-'],
          { cwd, env },
          (error, stdout, stderr) => {
            // Process any errors/warnings during formatting. These aren't all
            // necessarily fatal, so this doesn't prevent edits from being
            // applied.
            if (error) {
              maxSDKManager.logger.error(`Formatting error:\n${stderr}`);
              reject(error);
              return;
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
              document.lineAt(
                document.lineCount - 1,
              ).rangeIncludingLineBreak.end,
            );
            resolve([new vscode.TextEdit(documentRange, stdout)]);
          },
        );

        process.stdin?.write(originalDocumentText);
        process.stdin?.end();
      });
    },
  });
}
