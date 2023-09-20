//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as os from 'os';
import * as path from 'path';
import * as process from 'process';
import * as vscode from 'vscode';

/**
 * Substitute the given string with some common VSCode variables.
 *
 * The full list of variable is available in
 * https://code.visualstudio.com/docs/editor/variables-reference but we only
 * process the ones that don't depend on the open documents or the task runner.
 */
export function substituteVariables(
    text: string, workspaceFolder: vscode.WorkspaceFolder|undefined) {
  text =
      text.replace(/\${workspaceFolder}/g, workspaceFolder?.uri.fsPath || "");
  text = text.replace(/\${workspaceFolderBasename}/g,
                      path.basename(workspaceFolder?.uri.fsPath || ""));
  text = text.replace(/\${userHome}/g, os.homedir());
  text = text.replace(/\${pathSeparator}/g, path.sep);

  while (true) {
    const match = text.match(/\${env:([^}]*)}/);
    if (!match)
      break;
    text = text.replace(new RegExp(`\\\${env:${match[1]}}`, "g"),
                        process.env[match[1]] || "");
  }
  return text;
}
