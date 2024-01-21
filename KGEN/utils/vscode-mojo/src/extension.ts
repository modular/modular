//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import {LoggingService} from './logging';
import {MOJOContext} from './mojoContext';

let loggingService: LoggingService;
let mojoContext: MOJOContext;

/**
 *  This method is called when the extension is activated. See the
 * `activationEvents` in the package.json file for the current events that
 * activate this extension.
 */
export function activate(context: vscode.ExtensionContext) {
  loggingService = new LoggingService('Mojo');
  loggingService.logInfo("Initializing the Mojo extension.")
  mojoContext = new MOJOContext(context, loggingService);
  mojoContext.activate();
  loggingService.logInfo("Mojo extension initialized.")
}

/**
 * This method is called with VS Code deactivates this extension because of
 * an upgrade, a window reload, the editor is shutting down, or the user
 * disabled the extension manually.
 */
export function deactivate() {
  loggingService.logInfo("Deactivating extension.");
  mojoContext.dispose();
  loggingService.dispose();
}
