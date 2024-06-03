//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

/**
 * Returns if the given extension context is a nightly build.
 */
export function isNightlyExtension(context: vscode.ExtensionContext) {
  return context.extension.id.endsWith('-nightly');
}
