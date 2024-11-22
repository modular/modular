//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as path from 'path';
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
  relativePath: string;
  baseName: string;

  constructor(uri: vscode.Uri) {
    this.uri = uri;
    this.baseName = path.basename(uri.fsPath);
    this.relativePath = vscode.workspace.asRelativePath(
      this.uri,
      /*includeWorkspaceFolder=*/ true,
    );
  }
}

export function isMojoFile(uri: Optional<vscode.Uri>): boolean {
  return (
    uri !== undefined &&
    (uri.fsPath.endsWith('.mojo') || uri.fsPath.endsWith('.🔥'))
  );
}

/**
 * @returns All the currently open Mojo files as tuple, where the first element
 *     is the active document if it's a mojo file, and the second element are
 *     all other mojo files in no particular order.
 */
export function getAllOpenMojoFiles(): [
  Optional<WorkspaceAwareFile>,
  WorkspaceAwareFile[],
] {
  const activeRawUri = vscode.window.activeTextEditor?.document.uri;
  const activeFile =
    activeRawUri && isMojoFile(activeRawUri)
      ? new WorkspaceAwareFile(activeRawUri)
      : undefined;

  let otherOpenFiles = vscode.window.tabGroups.all
    .flatMap((tabGroup) => tabGroup.tabs)
    .map((tab) => (tab.input as any)?.uri)
    .filter(isMojoFile)
    .map((uri) => new WorkspaceAwareFile(uri))
    // We remove the active file from this list.
    .filter(
      (file) => !activeFile || file.uri.toString() != activeFile.uri.toString(),
    );

  return [activeFile, otherOpenFiles];
}

export async function directoryExists(path: string): Promise<boolean> {
  try {
    const stat = await vscode.workspace.fs.stat(vscode.Uri.file(path));
    if (stat.type & vscode.FileType.Directory) {
      return true;
    }
  } catch (e) {}
  return false;
}

export async function fileExists(path: string): Promise<boolean> {
  try {
    const stat = await vscode.workspace.fs.stat(vscode.Uri.file(path));
    if (stat.type & (vscode.FileType.File | vscode.FileType.SymbolicLink)) {
      return true;
    }
  } catch (e) {}
  return false;
}

export async function readFile(path: string): Promise<Optional<string>> {
  try {
    return new TextDecoder().decode(
      await vscode.workspace.fs.readFile(vscode.Uri.file(path)),
    );
  } catch {
    return undefined;
  }
}

export async function writeFile(
  path: string,
  contents: string,
): Promise<boolean> {
  try {
    await vscode.workspace.fs.writeFile(
      vscode.Uri.file(path),
      new TextEncoder().encode(contents),
    );
    return true;
  } catch {
    return false;
  }
}

export async function moveUpUntil(
  fsPath: string,
  condition: (p: string) => Promise<boolean>,
): Promise<Optional<string>> {
  while (fsPath.length > 0) {
    if (await condition(fsPath)) {
      return fsPath;
    }
    const dirname = path.dirname(fsPath);
    if (dirname === fsPath) {
      break;
    }
    fsPath = dirname;
  }
  return undefined;
}
