//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import {MOJOContext} from './mojoContext';

/**
 *  This method is called when the extension is activated. The extension is
 *  activated the very first time a command is executed.
 */
export function activate(context: vscode.ExtensionContext) {
  const outputChannel = vscode.window.createOutputChannel('Mojo');
  context.subscriptions.push(outputChannel);

  // Initialize the Mojo context.
  const mojoContext = new MOJOContext();
  context.subscriptions.push(mojoContext);

  // Initialize the commands of the extension.
  context.subscriptions.push(
      vscode.commands.registerCommand('mojo.restart', async () => {
        // Dispose and reactivate the context.
        mojoContext.dispose();
        await mojoContext.activate(outputChannel);
      }));
  context.subscriptions.push(
      vscode.commands.registerCommand('mojo.restart-suspended', async () => {
        // Dispose and reactivate the context.
        mojoContext.dispose();
        await mojoContext.activate(outputChannel, /*launchSuspended=*/ true);
      }));

  mojoContext.activate(outputChannel);
}
