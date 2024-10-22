//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

/**
 *  Gets the config value `mojo.<key>`, with an optional workspace folder.
 */
export function get<T>(
  key: string,
  workspaceFolder: Optional<vscode.WorkspaceFolder>,
): Optional<T>;

/**
 *  Gets the config value `mojo.<key>`, with an optional workspace folder and a
 * default value.
 */
export function get<T>(
  key: string,
  workspaceFolder: Optional<vscode.WorkspaceFolder>,
  defaultValue: T,
): T;

export function get<T>(
  key: string,
  workspaceFolder: Optional<vscode.WorkspaceFolder> = undefined,
  defaultValue: Optional<T> = undefined,
): Optional<T> {
  if (defaultValue === undefined) {
    return vscode.workspace
      .getConfiguration('mojo', workspaceFolder)
      .get<T>(key);
  }
  return vscode.workspace
    .getConfiguration('mojo', workspaceFolder)
    .get<T>(key, defaultValue);
}

/**
 *  Sets the config value `mojo.<key>`.
 */
export function update<T>(
  key: string,
  value: T,
  target?: vscode.ConfigurationTarget,
) {
  return vscode.workspace.getConfiguration('mojo').update(key, value, target);
}
