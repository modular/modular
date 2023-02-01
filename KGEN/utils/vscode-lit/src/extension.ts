//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import {registerFormatter} from './formatter';
import {LITContext} from './litContext';

/**
 *  This method is called when the extension is activated. The extension is
 *  activated the very first time a command is executed.
 */
export function activate(context: vscode.ExtensionContext) {
  const outputChannel = vscode.window.createOutputChannel('Lit');
  context.subscriptions.push(outputChannel);

  // Initialize the formatter.
  context.subscriptions.push(registerFormatter(outputChannel));

  // Initialize the LIT context.
  const litContext = new LITContext();
  context.subscriptions.push(litContext);

  // Initialize the commands of the extension.
  context.subscriptions.push(
      vscode.commands.registerCommand('lit.restart', async () => {
        // Dispose and reactivate the context.
        litContext.dispose();
        await litContext.activate(outputChannel);
      }));

  litContext.activate(outputChannel);
}
