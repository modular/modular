//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as path from 'path';
import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient/node';
import {TransportKind} from 'vscode-languageclient/node';

import {InitializationOptions} from '../../lsp-proxy/src/types';
import {MojoContext} from "../mojoContext";
import * as config from '../utils/config';
import {DisposableContext} from "../utils/disposableContext";

/**
 *  This class manages the LSP clients.
 */
export class MojoLSPContext extends DisposableContext {
  private mojoContext: MojoContext;
  private lspClients: Map<string, Promise<vscodelc.LanguageClient|undefined>> =
      new Map();

  constructor(mojoContext: MojoContext) {
    super();

    this.mojoContext = mojoContext;
  }

  async activate(launchServerWithDebuggerAttached: boolean = false) {
    this.pushSubscription(vscode.commands.registerCommand(
        'mojo.restart-and-debug-lsp', async () => {
          this.dispose();
          await this.activate(/*launchServerWithDebuggerAttached=*/ true);
        }));
    this.pushSubscription(
        vscode.commands.registerCommand('mojo.restart-lsp', async () => {
          this.dispose();
          await this.activate();
        }));

    // This lambda is used to lazily start language clients for the given
    // document. It removes the need to pro-actively start language clients for
    // every folder within the workspace.
    const startClientOnOpenDocument = async (document: vscode.TextDocument) => {
      this.getOrActivateLanguageClient(document.uri,
                                       launchServerWithDebuggerAttached);
    };
    // Watch any new documents to spawn servers when necessary.
    this.pushSubscription(
        vscode.workspace.onDidOpenTextDocument(startClientOnOpenDocument));

    // Process any existing documents.
    await Promise.all(vscode.workspace.textDocuments.map(
        doc => startClientOnOpenDocument(doc)));
  }

  /**
   * Open or return a language server for the given uri and language.
   */
  async getOrActivateLanguageClient(uri: vscode.Uri,
                                    launchServerWithDebuggerAttached: boolean):
      Promise<vscodelc.LanguageClient|undefined> {
    if (!uri.fsPath.endsWith(".mojo") && !uri.fsPath.endsWith('🔥') &&
        !uri.fsPath.endsWith(".ipynb"))
      return undefined;

    this.mojoContext.loggingService.lsp.logInfo(
        `Activating language client for URI '${uri}'`)
    // Check the scheme of the uri.
    let validSchemes = [ 'file', 'vscode-notebook-cell' ];
    if (!validSchemes.includes(uri.scheme)) {
      this.mojoContext.loggingService.lsp.logInfo(
          `Unsupported URI scheme '${uri.scheme}'`)
      return undefined;
    }

    // Resolve the workspace folder if this document is in one. We use the
    // workspace folder when determining if a server needs to be started.

    let workspaceFolder = vscode.workspace.getWorkspaceFolder(uri);
    let cacheKey =
        workspaceFolder ? workspaceFolder.uri.toString() : uri.fsPath;

    // Get or create a client context for this folder.
    if (!this.lspClients.has(cacheKey)) {
      this.lspClients.set(
          cacheKey, this.startLanguageClient(workspaceFolder,
                                             launchServerWithDebuggerAttached));
    }
    return this.lspClients.get(cacheKey);
  }

  /**
   *  Start a new language client.
   */
  async startLanguageClient(workspaceFolder: vscode.WorkspaceFolder|undefined,
                            launchServerWithDebuggerAttached: boolean):
      Promise<vscodelc.LanguageClient|undefined> {
    this.mojoContext.loggingService.lsp.logInfo(
        `Starting language client for workspace: ${
            workspaceFolder?.uri.fsPath}`);
    const clientTitle = 'Mojo Language Client';

    // Get the path of the lsp-server that is used to provide language
    // functionality.
    let sdk = await this.mojoContext.sdkManager.findSDK();
    if (!sdk)
      return undefined;

    let serverArgs: string[] = [];
    if (launchServerWithDebuggerAttached)
      serverArgs.push("--attach-debugger-on-startup");

    const includeDirs = await config.get<string[]|undefined>("lsp.includeDirs",
                                                             workspaceFolder) ||
                        [];
    for (const includeDir of includeDirs)
      serverArgs.push("-I", includeDir);

    const initializationOptions: InitializationOptions = {
      serverArgs : serverArgs,
      serverEnv : sdk.config.getProcessEnv(),
      serverPath : sdk.config.mojoLanguageServerPath,
    };

    const module = this.mojoContext.extensionContext.asAbsolutePath(
        path.join('lsp-proxy', 'out', 'proxy.js'));
    const serverOptions: vscodelc.ServerOptions = {
      run : {module, transport : TransportKind.ipc},
      debug : {module, transport : TransportKind.ipc}
    };

    // Configure file patterns relative to the workspace folder.
    let filePattern: vscode.GlobPattern = '**/*.{mojo,🔥,ipynb}';
    let selectorPattern: string|undefined = undefined;
    if (workspaceFolder) {
      filePattern = new vscode.RelativePattern(workspaceFolder, filePattern);
      selectorPattern = `${workspaceFolder.uri.fsPath}/**/*`;
    }

    // Configure the middleware of the client. This is sort of abused to allow
    // for defining a "fallback" language server that operates on non-workspace
    // folders. Workspace folder language servers can properly filter out
    // documents not within the folder, but we can't effectively filter for
    // documents outside of the workspace. To support this, and avoid having two
    // servers targeting the same set of files, we use middleware to inject the
    // dynamic logic for checking if a document is in the workspace.
    let middleware = {};
    if (!workspaceFolder) {
      middleware = {
        didOpen : (document: any, next: any) : Promise<void> => {
          if (!vscode.workspace.getWorkspaceFolder(document.uri)) {
            return next(document);
          }
          return Promise.resolve();
        }
      };
    }

    // Configure the client options.
    const clientOptions: vscodelc.LanguageClientOptions = {
      documentSelector : [
        {
          language : 'mojo',
          pattern : selectorPattern,
        },
        {
          scheme : "vscode-notebook-cell",
          language : "mojo",
          pattern : selectorPattern,
        },
      ],
      synchronize : {
        // Notify the server about file changes to language files contained in
        // the workspace.
        fileEvents : vscode.workspace.createFileSystemWatcher(filePattern)
      },
      outputChannel : this.mojoContext.loggingService.lsp.outputChannel,
      workspaceFolder : workspaceFolder,
      middleware : middleware,

      // Don't switch to output window when the server returns output.
      revealOutputChannelOn : vscodelc.RevealOutputChannelOn.Never,
      initializationOptions : initializationOptions,
    };

    // Create the language client and start the client.
    let languageClient = new vscodelc.LanguageClient(
        'mojo-lsp', clientTitle, serverOptions, clientOptions);
    this.mojoContext.loggingService.lsp.logInfo(
        `Launching Language Server '${
            initializationOptions.serverPath}' with options:`,
        initializationOptions.serverArgs);
    this.mojoContext.loggingService.lsp.logInfo("Launching Language Server");
    languageClient.start();
    return languageClient;
  }

  public dispose() {
    super.dispose();
    const clients = [...this.lspClients.entries() ];
    this.lspClients.clear();

    Promise.all(clients.map(async ([ key, client ]) => {
      const resolvedClient = await client;
      if (resolvedClient) {
        this.mojoContext.loggingService.lsp.logInfo(
            `Stopping Language Server for URI '${key}'`);
        resolvedClient.stop();
        resolvedClient.dispose();
      }
    }));
  }
}
