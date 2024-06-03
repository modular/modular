//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import {LoggingService} from './logging';
import {MojoContext} from './mojoContext';
import {isNightlyExtension} from './utils/buildInfo';

/**
 * This class provides an entry point for the Mojo extension, managing the
 * extension's state and disposal.
 */
class MojoExtension {
  private readonly mojoContext: MojoContext;
  private readonly loggingService: LoggingService;

  constructor(context: vscode.ExtensionContext) {
    const isNightly = isNightlyExtension(context);
    this.loggingService = new LoggingService(isNightly);
    this.loggingService.main.logInfo("Initializing the Mojo extension.");
    this.mojoContext = new MojoContext(context, this.loggingService);
    this.mojoContext.activate();
    this.loggingService.main.logInfo("Mojo extension initialized.");

    // Check and warn for incompatible extensions.
    this.checkForIncompatibleExtensions(isNightly);
  }

  async checkForIncompatibleExtensions(isNightly: boolean) {
    const stableExtensionId = 'modular-mojotools.vscode-mojo';
    const nightlyExtensionId = 'modular-mojotools.vscode-mojo-nightly';

    // Only one Mojo extension can be active at any given time, and intermixing
    // them can lead to unexpected behavior. If this is a stable extension,
    // check for a nightly extension, and vice versa.
    const invalidExtension = vscode.extensions.getExtension(
        isNightly ? stableExtensionId : nightlyExtensionId);
    if (!invalidExtension)
      return;

    vscode.window
        .showWarningMessage(
            'You have both the stable and nightly versions of the Mojo ' +
                'extension enabled. Please disable one of them to avoid ' +
                'conflicts.',
            'Show Extensions')
        .then((value) => {
          if (value === 'Show Extensions') {
            vscode.commands.executeCommand('workbench.extensions.search',
                                           "@id:" + stableExtensionId + " " +
                                               "@id:" + nightlyExtensionId);
          }
        });
  }

  public dispose() {
    this.loggingService.main.logInfo("Deactivating extension.");
    this.mojoContext.dispose();
    this.loggingService.dispose();
  }
}

let extension: MojoExtension;

/**
 *  This method is called when the extension is activated. See the
 * `activationEvents` in the package.json file for the current events that
 * activate this extension.
 */
export function activate(context: vscode.ExtensionContext) {
  extension = new MojoExtension(context);
}

/**
 * This method is called with VS Code deactivates this extension because of
 * an upgrade, a window reload, the editor is shutting down, or the user
 * disabled the extension manually.
 */
export function deactivate() { extension.dispose(); }
