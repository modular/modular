//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

/**
 *  Gets the config value `lit.<key>`, with an optional workspace folder.
 */
export function get<T>(key: string,
                       workspaceFolder: vscode.WorkspaceFolder = null,
                       defaultValue: T = undefined): T {
  return vscode.workspace.getConfiguration('lit', workspaceFolder)
      .get<T>(key, defaultValue);
}

/**
 *  Sets the config value `lit.<key>`.
 */
export function update<T>(key: string, value: T,
                          target?: vscode.ConfigurationTarget) {
  return vscode.workspace.getConfiguration('lit').update(key, value, target);
}
