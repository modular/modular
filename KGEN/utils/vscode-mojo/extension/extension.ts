//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import { Logger } from './logging';
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
import * as config from './utils/config';
import * as configWatcher from './utils/configWatcher';
import { MojoSDKSpec } from './sdk/types';

/**
 * This class provides an entry point for the Mojo extension, managing the
 * extension's state and disposal.
 */
export class MojoExtension extends DisposableContext {
  public logger?: Logger;
  public readonly extensionContext: vscode.ExtensionContext;
  public lspManager?: MojoLSPManager;

  constructor(context: vscode.ExtensionContext) {
    super();
    this.extensionContext = context;
  }

  async activate(initializationSDK?: Optional<MojoSDKSpec>) {
    const isNightly = isNightlyExtension(this.extensionContext);
    const logger = new Logger(isNightly);
    this.logger = logger;

    if (this.areThereIncompatibleExtensions(isNightly)) {
      this.logger.main.logInfo(
        'Not activating the Mojo Context due to another Mojo extension being enabled.'
      );
      return;
    }

    this.logger.main.logInfo('Activating the Mojo Context.');

    const enableMagicSDK = config.get<boolean>(
      'enableMagicSDK',
      /*workspaceFolder=*/ undefined,
      false
    );
    this.pushSubscription(
      await configWatcher.activate(
        /*workspaceFolder=*/ undefined,
        ['enableMagicSDK'],
        /*paths=*/ []
      )
    );
    const sdkManager = new MojoSDKManager(
      this.logger,
      this.extensionContext,
      initializationSDK,
      enableMagicSDK
    );
    this.pushSubscription(sdkManager);

    // Initialize the commands of the extension.
    this.pushSubscription(
      vscode.commands.registerCommand(
        'mojo.restart',
        async (initializationSDK: Optional<MojoSDKSpec>) => {
          // Dispose and reactivate the context.
          this.dispose();
          await this.activate(initializationSDK);
        }
      )
    );

    // Initialize the testing support.
    let testManager = new MojoTestManager(sdkManager);
    await testManager.activate();
    this.pushSubscription(testManager);

    // Initialize the formatter.
    this.pushSubscription(registerFormatter(sdkManager));

    // Initialize the debugger support.
    this.pushSubscription(new MojoDebugManager(this, sdkManager));

    // Initialize the execution commands.
    this.pushSubscription(activateRunCommands(sdkManager));

    // Initialize the decorations.
    this.pushSubscription(new MojoDecoratorManager());

    // Initialize the LSPs
    this.lspManager = new MojoLSPManager(sdkManager, this.extensionContext);
    await this.lspManager.activate();
    this.pushSubscription(this.lspManager);

    this.logger.main.logInfo('MojoContext activated.');
    this.pushSubscription(
      new vscode.Disposable(() => {
        logger.main.logInfo('Disposing MOJOContext.');
      })
    );

    // Initialize the RPC server
    const rpcServer = new RpcServer(this.logger);
    this.logger.main.logInfo('Starting RPC server');
    this.pushSubscription(rpcServer);
    rpcServer.listen();
    this.logger.main.logInfo('Mojo extension initialized.');
  }

  private areThereIncompatibleExtensions(isNightly: boolean): boolean {
    const stableExtensionId = 'modular-mojotools.vscode-mojo';
    const nightlyExtensionId = 'modular-mojotools.vscode-mojo-nightly';

    // Only one Mojo extension can be active at any given time, and intermixing
    // them can lead to unexpected behavior. If this is a stable extension,
    // check for a nightly extension, and vice versa.
    const invalidExtension = vscode.extensions.getExtension(
      isNightly ? stableExtensionId : nightlyExtensionId
    );

    if (!invalidExtension) {
      return false;
    }

    vscode.window
      .showWarningMessage(
        'You have both the stable and nightly versions of the Mojo ' +
          'extension enabled. Please disable one of them to avoid ' +
          'conflicts and then restart the editor.',
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
    return true;
  }

  override dispose() {
    this.logger?.main.logInfo('Deactivating extension.');
    super.dispose();
    this.logger?.dispose();
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
    extension.dispose();
  });
}

export function getExtension(): Promise<MojoExtension> {
  return extension;
}
