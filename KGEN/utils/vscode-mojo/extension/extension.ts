//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import { Logger } from './logging';
import { MojoSDKManager } from './sdk/sdkManager';
import { MojoLSPManager } from './lsp/lsp';
import { DisposableContext } from './utils/disposableContext';
import { MojoTestManager } from './testing/testing';
import { registerFormatter } from './formatter';
import { activateRunCommands } from './commands/run';
import { MojoDebugManager } from './debug/debug';
import { MojoDecoratorManager } from './decorations';
import { RpcServer } from './server/RpcServer';
import { MojoSDKSpec } from './sdk/types';
import { Mutex } from 'async-mutex';

/**
 * State that survives across reloads of the extension, but not re-activations.
 */
export class ExtensionSemiPersistentState {
  public seenDevSDKs = new Set<string>();
}

/**
 * Returns if the given extension context is a nightly build.
 */
export function isNightlyExtension(context: vscode.ExtensionContext) {
  return context.extension.id.endsWith('-nightly');
}

/**
 * This class provides an entry point for the Mojo extension, managing the
 * extension's state and disposal.
 *
 * The MojoExtension class and its components don't really have dynamic
 * states. Instead, when a major configuration changes, the extension restarts
 * completely with the new configuration. This can be seen, for example,
 * when selecting the SDK upon initialization: once the initial SDK is
 * selected, a full restart happens with that SDK forced as part of the
 * new initialization. This approach simplifies greatly the architecture of
 * the code and can keep us away from redux-like workflows, which are great,
 * but not worth the price at this point.
 */
export class MojoExtension extends DisposableContext {
  public logger: Logger;
  public readonly extensionContext: vscode.ExtensionContext;
  public lspManager?: MojoLSPManager;
  public readonly isNightly: boolean;
  private semiPersistentState = new ExtensionSemiPersistentState();
  private activateMutex = new Mutex();

  constructor(
    context: vscode.ExtensionContext,
    logger: Logger,
    isNightly: boolean
  ) {
    super();
    this.extensionContext = context;
    this.logger = logger;
    this.isNightly = isNightly;
  }

  async activate(
    initializationSDK: Optional<MojoSDKSpec>,
    reloading: boolean
  ): Promise<MojoExtension> {
    return await this.activateMutex.runExclusive(async () => {
      if (reloading) {
        this.dispose();
      }

      if (this.areThereIncompatibleExtensions(this.isNightly)) {
        this.logger.main.logInfo(
          'Not activating the Mojo Context due to another Mojo extension being enabled.'
        );
        return this;
      }

      this.logger.main.logInfo(`
=============================
Activating the Mojo Extension
=============================
`);

      const sdkManager = new MojoSDKManager(
        this.logger,
        this.extensionContext,
        initializationSDK,
        this.isNightly,
        this.semiPersistentState
      );
      this.pushSubscription(sdkManager);

      // Initialize the restart command, which can optionally receive an
      // initialization SDK to force the extension to use it without
      // performing any SDK fetching work.
      this.pushSubscription(
        vscode.commands.registerCommand(
          'mojo.extension.restart',
          async (initializationSDK: Optional<MojoSDKSpec>) => {
            // Dispose and reactivate the context.
            await this.activate(initializationSDK, /*reloading=*/ true);
          }
        )
      );

      // Initialize the testing support.
      let testManager = new MojoTestManager(sdkManager, this.logger);
      await testManager.activate();
      this.pushSubscription(testManager);

      // Initialize the formatter.
      this.pushSubscription(registerFormatter(sdkManager));

      // Initialize the debugger support.
      this.pushSubscription(new MojoDebugManager(this, sdkManager));

      // Initialize the execution commands.
      this.pushSubscription(
        activateRunCommands(sdkManager, this.extensionContext)
      );

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
      return this;
    });
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
    this.logger.main.logInfo('Disposing the extension.');
    super.dispose();
  }
}

export let extension: MojoExtension;
let logger: Logger;
let logHook: (level: string, message: string) => void;

/**
 *  This method is called when the extension is activated. See the
 * `activationEvents` in the package.json file for the current events that
 * activate this extension.
 */
export function activate(
  context: vscode.ExtensionContext
): Promise<MojoExtension> {
  const isNightly = isNightlyExtension(context);
  logger = new Logger(isNightly);

  if (logHook) {
    logger.main.logCallback = logHook;
    logger.lsp.logCallback = logHook;
  }

  extension = new MojoExtension(context, logger, isNightly);
  return extension.activate(
    /*initializationSDK=*/ undefined,
    /*reloading=*/ false
  );
}

/**
 * This method is called with VS Code deactivates this extension because of
 * an upgrade, a window reload, the editor is shutting down, or the user
 * disabled the extension manually.
 */
export function deactivate() {
  logger.main.logInfo('Deactivating the extension.');
  extension.dispose();
  logger.main.logInfo('Extension deactivated.');
  logger.dispose();
}

export function setLogHook(hook: (level: string, message: string) => void) {
  logHook = hook;
  if (logger) {
    logger.main.logCallback = hook;
    logger.lsp.logCallback = hook;
  }
}
