//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import { activateRunCommands } from './commands/run';
import { MojoDebugContext } from './debug/debug';
import { MojoDecoratorContext } from './decorations';
import { registerFormatter } from './formatter';
import { LoggingService } from './logging';
import { MojoLSPContext } from './lsp/lsp';
import { MojoSDKManager } from './mojoSDK';
import { MojoTestContext } from './testing/testing';
import { DisposableContext } from './utils/disposableContext';

/**
 *  This class manages the Mojo extension state.
 */
export class MojoContext extends DisposableContext {
  readonly sdkManager: MojoSDKManager;
  readonly loggingService: LoggingService;
  readonly extensionContext: vscode.ExtensionContext;
  lspContext?: MojoLSPContext;

  constructor(
    extensionContext: vscode.ExtensionContext,
    loggingService: LoggingService
  ) {
    super();
    this.extensionContext = extensionContext;
    this.loggingService = loggingService;
    this.sdkManager = new MojoSDKManager(this.loggingService, extensionContext);
  }

  /**
   *  Activate the Mojo context, and start the language clients.
   */
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
    let testContext = new MojoTestContext(this);
    await testContext.activate();
    this.pushSubscription(testContext);

    // Initialize the formatter.
    this.pushSubscription(
      registerFormatter(this.loggingService, this.sdkManager)
    );

    // Initialize the debugger support.
    this.pushSubscription(new MojoDebugContext(this));

    // Initialize the execution commands.
    this.pushSubscription(activateRunCommands(this));

    // Initialize the decorations.
    this.pushSubscription(new MojoDecoratorContext());

    // Initialize the LSPs
    this.lspContext = new MojoLSPContext(this);
    await this.lspContext.activate();
    this.pushSubscription(this.lspContext);

    this.loggingService.main.logInfo('MojoContext activated.');
    this.pushSubscription(
      new vscode.Disposable(() => {
        this.loggingService.main.logInfo('Disposing MOJOContext.');
      })
    );
  }
}
