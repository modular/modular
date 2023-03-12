//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

/**
 *  Gets the config value `mojo.<key>`, with an optional workspace folder.
 */
export function get<T>(key: string,
                       workspaceFolder: vscode.WorkspaceFolder = null,
                       defaultValue: T = undefined): T {
  return vscode.workspace.getConfiguration('mojo', workspaceFolder)
      .get<T>(key, defaultValue);
}

/**
 *  Sets the config value `mojo.<key>`.
 */
export function update<T>(key: string, value: T,
                          target?: vscode.ConfigurationTarget) {
  return vscode.workspace.getConfiguration('mojo').update(key, value, target);
}
