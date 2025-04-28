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
import { MAXSDK } from '../sdk/sdk';
import * as config from '../utils/config';
import { DisposableContext } from '../utils/disposableContext';
import { Subject } from 'rxjs';
import { Logger } from '../logging';
import { MAXSDKManager } from '../sdk/sdkManager';
import { TelemetryReporter } from '../telemetry';
import { LSPRecorder } from './recorder';

/**
 *  This class manages the LSP clients.
 */
export class MojoLSPManager extends DisposableContext {
  private sdkManager: MAXSDKManager;
  private extensionContext: vscode.ExtensionContext;
  public lspClient: Optional<vscodelc.LanguageClient>;
  public lspClientChanges = new Subject<Optional<vscodelc.LanguageClient>>();
  private logger: Logger;
  private reporter: TelemetryReporter;
  private recorder: Optional<LSPRecorder>;
  private statusBarItem: Optional<vscode.StatusBarItem>;
  private attachDebugger: boolean = false;

  constructor(
    sdkManager: MAXSDKManager,
    extensionContext: vscode.ExtensionContext,
    reporter: TelemetryReporter,
  ) {
    super();

    this.sdkManager = sdkManager;
    this.extensionContext = extensionContext;
    this.logger = sdkManager.logger;
    this.reporter = reporter;
  }

  async activate() {
    this.pushSubscription(
      vscode.commands.registerCommand('mojo.lsp.restart', async () => {
        // Wait for the language server to stop. This allows a graceful shutdown of the server instead of simply terminating the process, which is important for tracing.
        if (this.lspClient) {
          await this.lspClient.stop();
        }

        this.dispose();
        this.lspClient = undefined;
        await this.activate();
      }),
    );

    if (
      this.extensionContext.extensionMode == vscode.ExtensionMode.Development
    ) {
      this.pushSubscription(
        vscode.commands.registerCommand('mojo.lsp.debug', async () => {
          if (this.lspClient) {
            await this.lspClient.stop();
          }

          this.attachDebugger = true;

          this.dispose();
          this.lspClient = undefined;
          await this.activate();
        }),
      );

      this.pushSubscription(
        vscode.commands.registerTextEditorCommand(
          'mojo.lsp.dumpParsedIR',
          async (textEditor) => {
            if (!this.lspClient) {
              return;
            }

            await this.lspClient.sendNotification('mojo/emitParsedIR', {
              uri: textEditor.document.uri.toString(),
            });
          },
        ),
      );
    }

    this.statusBarItem = vscode.window.createStatusBarItem(
      'lsp-recording-state',
      vscode.StatusBarAlignment.Right,
    );
    this.statusBarItem.text = 'Mojo LSP $(record)';
    this.statusBarItem.backgroundColor = new vscode.ThemeColor(
      'statusBarItem.warningBackground',
    );
    this.statusBarItem.command = 'mojo.lsp.stopRecord';
    this.pushSubscription(this.statusBarItem);

    this.pushSubscription(
      vscode.commands.registerCommand('mojo.lsp.startRecord', async () => {
        if (this.recorder) {
          this.recorder.dispose();
        }

        if (
          !vscode.workspace.workspaceFolders ||
          vscode.workspace.workspaceFolders.length == 0
        ) {
          return;
        }
        const workspaceFolder = vscode.workspace.workspaceFolders[0];
        const recordPath = vscode.Uri.joinPath(
          workspaceFolder.uri,
          'mojo-lsp-recording.jsonl',
        );

        this.recorder = new LSPRecorder(recordPath.fsPath);
        this.pushSubscription(this.recorder);

        vscode.window
          .showInformationMessage(
            `Started recording language server session to ${recordPath}.`,
            'Stop',
            'Open',
          )
          .then((action) => {
            switch (action) {
              case 'Open':
                return vscode.commands.executeCommand(
                  'vscode.open',
                  recordPath,
                );
              case 'Stop':
                return vscode.commands.executeCommand('mojo.lsp.stopRecord');
            }
          });

        this.statusBarItem!.tooltip = `Recording Mojo LSP session to ${recordPath}`;
        this.statusBarItem!.show();
      }),
    );

    this.pushSubscription(
      vscode.commands.registerCommand('mojo.lsp.stopRecord', async () => {
        if (!this.recorder) {
          return;
        }

        this.recorder!.dispose();
        this.recorder = undefined;
        this.statusBarItem!.hide();
      }),
    );

    vscode.workspace.textDocuments.forEach((doc) =>
      this.tryStartLanguageClient(doc),
    );
    this.pushSubscription(
      vscode.workspace.onDidOpenTextDocument((doc) =>
        this.tryStartLanguageClient(doc),
      ),
    );
    this.pushRxjsSubscription(
      this.sdkManager.onActiveSDKChanged.subscribe(() => {
        vscode.commands.executeCommand('mojo.lsp.restart');
      }),
    );
  }

  async tryStartLanguageClient(doc: vscode.TextDocument): Promise<void> {
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
      [],
    );
    const lspClient = this.activateLanguageClient(sdk, includeDirs);
    this.lspClient = lspClient;
    this.lspClientChanges.next(lspClient);
    this.pushSubscription(
      new vscode.Disposable(() => {
        lspClient.stop();
        lspClient.dispose();
        this.lspClientChanges.next(undefined);
        this.lspClientChanges.unsubscribe();
      }),
    );
  }

  /**
   * Create a new language server.
   */
  activateLanguageClient(
    sdk: MAXSDK,
    includeDirs: string[],
  ): vscodelc.LanguageClient {
    this.logger.lsp.info('Activating language client');

    let serverArgs: string[] = [];

    for (const includeDir of includeDirs) {
      serverArgs.push('-I', includeDir);
    }

    if (this.attachDebugger) {
      serverArgs.push('--attach-debugger-on-startup');
    }

    const initializationOptions: InitializationOptions = {
      serverArgs: serverArgs,
      serverEnv: sdk.getProcessEnv(),
      serverPath: sdk.config.mojoLanguageServerPath,
    };

    const module = this.extensionContext.asAbsolutePath(
      path.join('lsp-proxy', 'out', 'proxy.js'),
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
          '**/*.{mojo,🔥,ipynb}',
        ),
      },
      outputChannel: this.logger.lsp.outputChannel,

      // Don't switch to output window when the server returns output.
      revealOutputChannelOn: vscodelc.RevealOutputChannelOn.Never,
      initializationOptions: initializationOptions,
    };

    clientOptions.middleware = {
      sendRequest: (method, param, token, next) => {
        if (this.recorder) {
          return this.recorder.sendRequest(method, param, token, next);
        } else {
          return next(method, param, token);
        }
      },
      sendNotification: (method, next, param) => {
        if (this.recorder) {
          return this.recorder.sendNotification(method, next, param);
        } else {
          return next(method, param);
        }
      },
    };

    // Create the language client and start the client.
    let languageClient = new vscodelc.LanguageClient(
      'mojo-lsp',
      'Mojo Language Client',
      serverOptions,
      clientOptions,
    );

    // The proxy sends us a mojo/lspRestart notification when it restarts the
    // underlying language server. It's our job to pass that to the telemetry
    // backend.
    this.pushSubscription(
      languageClient.onNotification('mojo/lspRestart', () => {
        this.reporter.sendTelemetryEvent('lspRestart', {
          mojoSDKVersion: sdk.config.version.toString(),
          mojoSDKKind: sdk.kind,
        });
      }),
    );

    this.logger.lsp.info(
      `Launching Language Server '${
        initializationOptions.serverPath
      }' with options:`,
      initializationOptions.serverArgs,
    );
    this.logger.lsp.info('Launching Language Server');
    // We intentionally don't await the `start` so that we can cancelling it
    // during a long initialization, which can happen when in debug mode.
    languageClient.start();
    return languageClient;
  }
}
