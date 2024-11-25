//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import { execFile } from 'child_process';
import * as vscode from 'vscode';

import { MAXSDKManager } from './sdk/sdkManager';
import { get } from './utils/config';
import { readFile, writeFile } from './utils/files';

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

      const mblackPath = sdk.config.mojoMBlackPath;
      // We try to fix the exec invocation within mblack if needed.
      // There's currently an issue in which mblack has an internal
      // path that is not escaped and creates issues when white
      // spaces are present.
      // TODO(SI-668): remove this when SI-668 gets fixed.

      const contents = await readFile(mblackPath);
      if (contents !== undefined) {
        const newContents = contents.replace(
          /'''exec' (\/.*) "\$0" "\$@"/i,
          `'''exec' '$1' "\$0" "\$@"`,
        );
        await writeFile(mblackPath, newContents);
      }

      let env = sdk.getProcessEnv();

      return new Promise<vscode.TextEdit[]>(function (resolve, reject) {
        const originalDocumentText = document.getText();
        const process = execFile(
          sdk.config.mojoDriverPath,
          ['format', '--quiet', ...args, '-'],
          { cwd, env },
          (error, stdout, stderr) => {
            // Process any errors/warnings during formatting. These aren't all
            // necessarily fatal, so this doesn't prevent edits from being
            // applied.
            if (error) {
              maxSDKManager.logger.main.logError(
                `Formatting error:\n${stderr}`,
              );
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
