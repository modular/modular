//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

/**
 *  Gets the config value `mojo.<key>`, with an optional workspace folder.
 */
export function get<T>(key: string, workspaceFolder: vscode.WorkspaceFolder|
                                    undefined): T|undefined;

/**
 *  Gets the config value `mojo.<key>`, with an optional workspace folder and a
 * default value.
 */
export function get<T>(key: string,
                       workspaceFolder: vscode.WorkspaceFolder|undefined,
                       defaultValue: T): T;

export function get<T>(
    key: string, workspaceFolder: vscode.WorkspaceFolder|undefined = undefined,
    defaultValue: T|undefined = undefined): T|undefined {
  if (defaultValue === undefined)
    return vscode.workspace.getConfiguration('mojo', workspaceFolder)
        .get<T>(key);
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
