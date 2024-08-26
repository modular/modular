//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import { LoggingService } from './logging';
import { isNightlyExtension } from './utils/buildInfo';
import { MojoSDKManager } from './sdk/sdkManager';
import { MojoLSPManager } from './lsp/lsp';
import { DisposableContext } from './utils/disposableContext';
import { MojoTestManager } from './testing/testing';
import { registerFormatter } from './formatter';
import { activateRunCommands } from './commands/run';
import { MojoDebugManager } from './debug/debug';
import { MojoDecoratorManager } from './decorations';
import { RpcServer } from './server/RpcServer';

/**
 * This class provides an entry point for the Mojo extension, managing the
 * extension's state and disposal.
 */
export class MojoExtension extends DisposableContext {
  public readonly loggingService: LoggingService;
  public readonly sdkManager: MojoSDKManager;
  public readonly extensionContext: vscode.ExtensionContext;
  public lspManager?: MojoLSPManager;

  constructor(context: vscode.ExtensionContext) {
    super();
    const isNightly = isNightlyExtension(context);
    this.loggingService = new LoggingService(isNightly);
    this.pushSubscription(this.loggingService);
    this.extensionContext = context;
    this.sdkManager = new MojoSDKManager(this.loggingService, context);

    // Check and warn for incompatible extensions.
    this.checkForIncompatibleExtensions(isNightly);
  }

  async activate() {
    this.loggingService.main.logInfo('Activating the Mojo Context.');
    // Initialize the commands of the extension.
    this.pushSubscription(
      vscode.commands.registerCommand('mojo.restart', async () => {
        // Dispose and reactivate the context.
        this.dispose();
        await this.activate();
      })
    );

    // Initialize the testing support.
    let testManager = new MojoTestManager(this);
    await testManager.activate();
    this.pushSubscription(testManager);

    // Initialize the formatter.
    this.pushSubscription(
      registerFormatter(this.loggingService, this.sdkManager)
    );

    // Initialize the debugger support.
    this.pushSubscription(new MojoDebugManager(this));

    // Initialize the execution commands.
    this.pushSubscription(activateRunCommands(this));

    // Initialize the decorations.
    this.pushSubscription(new MojoDecoratorManager());

    // Initialize the LSPs
    this.lspManager = new MojoLSPManager(this);
    await this.lspManager.activate();
    this.pushSubscription(this.lspManager);

    this.loggingService.main.logInfo('MojoContext activated.');
    this.pushSubscription(
      new vscode.Disposable(() => {
        this.loggingService.main.logInfo('Disposing MOJOContext.');
      })
    );

    // Initialize the RPC server
    const rpcServer = new RpcServer(this.loggingService);
    this.loggingService.main.logInfo('Starting RPC server');
    this.pushSubscription(rpcServer);
    rpcServer.listen();
    this.loggingService.main.logInfo('Mojo extension initialized.');
  }

  async checkForIncompatibleExtensions(isNightly: boolean) {
    const stableExtensionId = 'modular-mojotools.vscode-mojo';
    const nightlyExtensionId = 'modular-mojotools.vscode-mojo-nightly';

    // Only one Mojo extension can be active at any given time, and intermixing
    // them can lead to unexpected behavior. If this is a stable extension,
    // check for a nightly extension, and vice versa.
    const invalidExtension = vscode.extensions.getExtension(
      isNightly ? stableExtensionId : nightlyExtensionId
    );

    if (!invalidExtension) {
      return;
    }

    vscode.window
      .showWarningMessage(
        'You have both the stable and nightly versions of the Mojo ' +
          'extension enabled. Please disable one of them to avoid ' +
          'conflicts.',
        'Show Extensions'
      )
      .then((value) => {
        if (value === 'Show Extensions') {
          vscode.commands.executeCommand(
            'workbench.extensions.search',
            '@id:' + stableExtensionId + ' ' + '@id:' + nightlyExtensionId
          );
        }
      });
  }
}

let extension: Promise<MojoExtension>;

/**
 *  This method is called when the extension is activated. See the
 * `activationEvents` in the package.json file for the current events that
 * activate this extension.
 */
export function activate(context: vscode.ExtensionContext) {
  let ext = new MojoExtension(context);

  extension = ext.activate().then(() => ext);
}

/**
 * This method is called with VS Code deactivates this extension because of
 * an upgrade, a window reload, the editor is shutting down, or the user
 * disabled the extension manually.
 */
export function deactivate() {
  extension.then((extension) => {
    extension.loggingService.main.logInfo('Deactivating extension.');
    extension.dispose();
  });
}

export function getExtension(): Promise<MojoExtension> {
  return extension;
}
