//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as path from "path";
import * as vscode from 'vscode';

/**
 * Utility class for handling files relative to their containing workspace
 * folder.
 */
export class WorkspaceAwareFile {
  uri: vscode.Uri;
  workspaceFolder?: vscode.WorkspaceFolder;
  /**
   * The path relative to its containing workspace folder, or the full file
   * system path if no workspace folder contains it. If it's a relative path, it
   * is prepended by the name of the workspace folder.
   */
  relativePath: string
  baseName: string;

  constructor(uri: vscode.Uri) {
    this.uri = uri;
    this.baseName = path.basename(uri.fsPath);
    this.relativePath = vscode.workspace.asRelativePath(
        this.uri, /*includeWorkspaceFolder=*/ true);
  }
}

/**
 * @returns All the currently open Mojo files as tuple, where the first element
 *     is the active document if it's a mojo file, and the second element are
 *     all other mojo files in no particular order.
 */
export function getAllOpenMojoFiles():
    [ WorkspaceAwareFile|undefined, WorkspaceAwareFile[] ] {
  const mojoUriFilter = (uri: vscode.Uri) =>
      uri && (uri.fsPath.endsWith(".mojo") || uri.fsPath.endsWith(".🔥"));

  const activeRawUri = vscode.window.activeTextEditor?.document.uri;
  const activeFile = activeRawUri && mojoUriFilter(activeRawUri)
                         ? new WorkspaceAwareFile(activeRawUri)
                         : undefined;

  let otherOpenFiles =
      vscode.window.tabGroups.all.flatMap(tabGroup => tabGroup.tabs)
          .map(tab => (tab.input as any)?.uri)
          .filter(mojoUriFilter)
          .map(uri => new WorkspaceAwareFile(uri))
          // We remove the active file from this list.
          .filter(file => !activeFile ||
                          file.uri.toString() != activeFile.uri.toString());

  return [ activeFile, otherOpenFiles ];
}
