//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as path from 'path';
import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient/node';
import { TransportKind } from 'vscode-languageclient/node';

import { InitializationOptions } from '../../lsp-proxy/src/types';
import { MojoSDK } from '../sdk/sdk';
import * as config from '../utils/config';
import { DisposableContext } from '../utils/disposableContext';
import { Subject } from 'rxjs';
import { Logger } from '../logging';
import { MojoSDKManager } from '../sdk/sdkManager';

/**
 *  This class manages the LSP clients.
 */
export class MojoLSPManager extends DisposableContext {
  private sdkManager: MojoSDKManager;
  private extensionContext: vscode.ExtensionContext;
  public lspClient: Optional<vscodelc.LanguageClient>;
  public lspClientChanges = new Subject<Optional<vscodelc.LanguageClient>>();
  private logger: Logger;

  constructor(
    sdkManager: MojoSDKManager,
    extensionContext: vscode.ExtensionContext
  ) {
    super();

    this.sdkManager = sdkManager;
    this.extensionContext = extensionContext;
    this.logger = sdkManager.logger;
  }

  async activate(launchServerWithDebuggerAttached: boolean = false) {
    this.pushSubscription(
      vscode.commands.registerCommand(
        'mojo.lsp.restart-and-debug',
        async () => {
          this.dispose();
          await this.activate(/*launchServerWithDebuggerAttached=*/ true);
        }
      )
    );
    this.pushSubscription(
      vscode.commands.registerCommand('mojo.lsp.restart', async () => {
        this.dispose();
        await this.activate();
      })
    );

    vscode.workspace.textDocuments.forEach((doc) =>
      this.tryStartLanguageClient(doc, launchServerWithDebuggerAttached)
    );
    this.pushSubscription(
      vscode.workspace.onDidOpenTextDocument((doc) =>
        this.tryStartLanguageClient(doc, launchServerWithDebuggerAttached)
      )
    );
  }

  async tryStartLanguageClient(
    doc: vscode.TextDocument,
    debuggerAttached: boolean
  ): Promise<void> {
    if (doc.languageId !== 'mojo') {
      return;
    }

    let sdk = await this.sdkManager.findSDK(/*hideRepeatedErrors=*/ true);

    if (!sdk) {
      return;
    }

    if (this.lspClient !== undefined) {
      return;
    }

    const includeDirs = config.get<string[]>(
      'lsp.includeDirs',
      /*workspaceFolder=*/ undefined,
      []
    );
    const lspClient = this.activateLanguageClient(
      debuggerAttached,
      sdk,
      includeDirs
    );
    this.lspClient = lspClient;
    this.lspClientChanges.next(lspClient);
    this.pushSubscription(
      new vscode.Disposable(() => {
        lspClient.stop();
        lspClient.dispose();
        this.lspClientChanges.next(undefined);
        this.lspClientChanges.unsubscribe();
      })
    );
  }

  /**
   * Create a new language server.
   */
  activateLanguageClient(
    launchServerWithDebuggerAttached: boolean,
    sdk: MojoSDK,
    includeDirs: string[]
  ): vscodelc.LanguageClient {
    this.logger.lsp.logInfo('Activating language client');

    let serverArgs: string[] = [];

    if (launchServerWithDebuggerAttached) {
      serverArgs.push('--attach-debugger-on-startup');
    }

    for (const includeDir of includeDirs) {
      serverArgs.push('-I', includeDir);
    }

    const initializationOptions: InitializationOptions = {
      serverArgs: serverArgs,
      serverEnv: sdk.getProcessEnv(),
      serverPath: sdk.config.mojoLanguageServerPath,
    };

    const module = this.extensionContext.asAbsolutePath(
      path.join('lsp-proxy', 'out', 'proxy.js')
    );
    const serverOptions: vscodelc.ServerOptions = {
      run: { module, transport: TransportKind.ipc },
      debug: { module, transport: TransportKind.ipc },
    };

    // Configure the client options.
    const clientOptions: vscodelc.LanguageClientOptions = {
      // The current selection mechanism indicates all documents to be served
      // by the same single LSP Server. This wouldn't work if at some point
      // we support multiple SDKs running at once, for which we'd need a more
      // flexible way to manage LSP Servers than `vscodelc`. Two options might
      // be feasible:
      // - Fork/contribute `vscodelc` and allow for a more customizable selection logic.
      // - Do the selection within the proxy, which would be "easy" to implement
      //   if the proxy is restarted with the new correct info whenever a new SDK is
      //   identified.
      documentSelector: [
        {
          language: 'mojo',
        },
        {
          scheme: 'vscode-notebook-cell',
          language: 'mojo',
        },
      ],
      synchronize: {
        // Notify the server about file changes following the given file
        // pattern.
        fileEvents: vscode.workspace.createFileSystemWatcher(
          '**/*.{mojo,🔥,ipynb}'
        ),
      },
      outputChannel: this.logger.lsp.outputChannel,

      // Don't switch to output window when the server returns output.
      revealOutputChannelOn: vscodelc.RevealOutputChannelOn.Never,
      initializationOptions: initializationOptions,
    };

    // Create the language client and start the client.
    let languageClient = new vscodelc.LanguageClient(
      'mojo-lsp',
      'Mojo Language Client',
      serverOptions,
      clientOptions
    );
    this.logger.lsp.logInfo(
      `Launching Language Server '${
        initializationOptions.serverPath
      }' with options:`,
      initializationOptions.serverArgs
    );
    this.logger.lsp.logInfo('Launching Language Server');
    // We intentionally don't await the `start` so that we can cancelling it
    // during a long initialization, which can happen when in debug mode.
    languageClient.start();
    return languageClient;
  }
}
