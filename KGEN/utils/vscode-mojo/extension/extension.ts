//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import { Logger } from './logging';
import { MAXSDKManager } from './sdk/sdkManager';
import { MojoLSPManager } from './lsp/lsp';
import * as configWatcher from './utils/configWatcher';
import { DisposableContext } from './utils/disposableContext';
import { registerFormatter } from './formatter';
import { activateRunCommands } from './commands/run';
import { MojoDebugManager } from './debug/debug';
import { MojoDecoratorManager } from './decorations';
import { RpcServer } from './server/RpcServer';
import { Mutex } from 'async-mutex';
import { TelemetryReporter } from './telemetry';

/**
 * Returns if the given extension context is a nightly build.
 */
export function isNightlyExtension(context: vscode.ExtensionContext) {
  return context.extension.id.endsWith('-nightly');
}

/**
 * This class provides an entry point for the Mojo extension, managing the
 * extension's state and disposal.
 */
export class MojoExtension extends DisposableContext {
  public logger: Logger;
  public readonly extensionContext: vscode.ExtensionContext;
  public lspManager?: MojoLSPManager;
  public readonly isNightly: boolean;
  private activateMutex = new Mutex();
  private reporter: TelemetryReporter;

  constructor(
    context: vscode.ExtensionContext,
    logger: Logger,
    isNightly: boolean,
  ) {
    super();
    this.extensionContext = context;
    this.logger = logger;
    this.isNightly = isNightly;
    // NOTE: The telemetry connection string comes from the Azure Application Insights dashboard.
    this.reporter = new TelemetryReporter(
      context.extension.packageJSON.telemetryConnectionString,
    );
    this.pushSubscription(this.reporter);

    // Disable telemetry for development and test environments.
    this.reporter.enabled =
      context.extensionMode == vscode.ExtensionMode.Production;
  }

  async activate(reloading: boolean): Promise<MojoExtension> {
    return await this.activateMutex.runExclusive(async () => {
      if (reloading) {
        this.dispose();
      }

      if (this.areThereIncompatibleExtensions(this.isNightly)) {
        this.logger.info(
          'Not activating the Mojo Context due to another Mojo extension being enabled.',
        );
        return this;
      }

      this.logger.info(`
=============================
Activating the Mojo Extension
=============================
`);

      const sdkManager = new MAXSDKManager(
        this.logger,
        this.isNightly,
        this.extensionContext,
      );
      this.pushSubscription(sdkManager);

      this.pushSubscription(
        await configWatcher.activate({
          settings: ['SDK.additionalSDKs'],
        }),
      );

      this.pushSubscription(
        vscode.commands.registerCommand('mojo.extension.restart', async () => {
          // Dispose and reactivate the context.
          await this.activate(/*reloading=*/ true);
        }),
      );

      // Initialize the formatter.
      this.pushSubscription(registerFormatter(sdkManager));

      // Initialize the debugger support.
      this.pushSubscription(new MojoDebugManager(this, sdkManager));

      // Initialize the execution commands.
      this.pushSubscription(
        activateRunCommands(sdkManager, this.extensionContext),
      );

      // Initialize the decorations.
      this.pushSubscription(new MojoDecoratorManager());

      // Initialize the LSPs
      this.lspManager = new MojoLSPManager(
        sdkManager,
        this.extensionContext,
        this.reporter,
      );
      await this.lspManager.activate();
      this.pushSubscription(this.lspManager);

      this.logger.info('MojoContext activated.');
      this.pushSubscription(
        new vscode.Disposable(() => {
          logger.info('Disposing MOJOContext.');
        }),
      );

      // Initialize the RPC server
      const rpcServer = new RpcServer(this.logger);
      this.logger.info('Starting RPC server');
      this.pushSubscription(rpcServer);
      rpcServer.listen();
      this.logger.info('Mojo extension initialized.');
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
      isNightly ? stableExtensionId : nightlyExtensionId,
    );

    if (!invalidExtension) {
      return false;
    }

    vscode.window
      .showWarningMessage(
        'You have both the stable and nightly versions of the Mojo ' +
          'extension enabled. Please disable one of them to avoid ' +
          'conflicts and then restart the editor.',
        'Show Extensions',
      )
      .then((value) => {
        if (value === 'Show Extensions') {
          vscode.commands.executeCommand(
            'workbench.extensions.search',
            '@id:' + stableExtensionId + ' ' + '@id:' + nightlyExtensionId,
          );
        }
      });
    return true;
  }

  override dispose() {
    this.logger.info('Disposing the extension.');
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
export async function activate(
  context: vscode.ExtensionContext,
): Promise<MojoExtension> {
  const isNightly = isNightlyExtension(context);
  logger = new Logger(isNightly);

  if (logHook) {
    logger.main.logCallback = logHook;
    logger.lsp.logCallback = logHook;
  }

  extension = new MojoExtension(context, logger, isNightly);
  return extension.activate(/*reloading=*/ false);
}

/**
 * This method is called with VS Code deactivates this extension because of
 * an upgrade, a window reload, the editor is shutting down, or the user
 * disabled the extension manually.
 */
export function deactivate() {
  logger.info('Deactivating the extension.');
  extension.dispose();
  logger.info('Extension deactivated.');
  logger.dispose();
}

export function setLogHook(hook: (level: string, message: string) => void) {
  logHook = hook;
  if (logger) {
    logger.main.logCallback = hook;
    logger.lsp.logCallback = hook;
  }
}
