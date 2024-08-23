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
import { MojoContext } from '../mojoContext';
import { MojoSDK } from '../sdk/sdk';
import * as config from '../utils/config';
import { DisposableContext } from '../utils/disposableContext';
import { Subject } from 'rxjs';

/**
 *  This class manages the LSP clients.
 */
export class MojoLSPContext extends DisposableContext {
  private mojoContext: MojoContext;
  public lspClient: vscodelc.LanguageClient | undefined;
  public lspClientChanges = new Subject<vscodelc.LanguageClient | undefined>();

  constructor(mojoContext: MojoContext) {
    super();

    this.mojoContext = mojoContext;
  }

  async activate(launchServerWithDebuggerAttached: boolean = false) {
    this.pushSubscription(
      vscode.commands.registerCommand(
        'mojo.restart-and-debug-lsp',
        async () => {
          this.dispose();
          await this.activate(/*launchServerWithDebuggerAttached=*/ true);
        }
      )
    );
    this.pushSubscription(
      vscode.commands.registerCommand('mojo.restart-lsp', async () => {
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

    let sdk = await this.mojoContext.sdkManager.findSDK();

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
    this.mojoContext.loggingService.lsp.logInfo('Activating language client');

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

    const module = this.mojoContext.extensionContext.asAbsolutePath(
      path.join('lsp-proxy', 'out', 'proxy.js')
    );
    const serverOptions: vscodelc.ServerOptions = {
      run: { module, transport: TransportKind.ipc },
      debug: { module, transport: TransportKind.ipc },
    };

    // Configure the client options.
    const clientOptions: vscodelc.LanguageClientOptions = {
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
      outputChannel: this.mojoContext.loggingService.lsp.outputChannel,

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
    this.mojoContext.loggingService.lsp.logInfo(
      `Launching Language Server '${
        initializationOptions.serverPath
      }' with options:`,
      initializationOptions.serverArgs
    );
    this.mojoContext.loggingService.lsp.logInfo('Launching Language Server');
    // We intentionally don't await the `start` so that we can cancelling it
    // during a long initialization, which can happen when in debug mode.
    languageClient.start();
    return languageClient;
  }
}
