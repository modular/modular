//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient/node';

import * as config from './config';
import * as configWatcher from './configWatcher';
import {registerFormatter} from './formatter';
import {LoggingService} from './logging';
import {MOJOSDK} from './mojoSDK';

/**
 *  This class manages the Mojo extension state, including the language
 *  client.
 */
export class MOJOContext implements vscode.Disposable {
  _sdk: MOJOSDK|undefined;
  subscriptions: vscode.Disposable[] = [];
  workspaceClients: Map<string, vscodelc.LanguageClient> = new Map();
  _loggingService: LoggingService|undefined;

  private getLoggingService(): LoggingService { return this._loggingService!; }

  private getSDK(): MOJOSDK { return this._sdk!; }

  /**
   *  Activate the Mojo context, and start the language clients.
   */
  async activate(loggingService: LoggingService,
                 launchLanguageServerSuspended: boolean = false) {
    this._loggingService = loggingService;
    this._sdk = new MOJOSDK(loggingService);

    // This lambda is used to lazily start language clients for the given
    // document. It removes the need to pro-actively start language clients for
    // every folder within the workspace.
    const startClientOnOpenDocument = async (document: vscode.TextDocument) => {
      await this.getOrActivateLanguageClient(document.uri,
                                             launchLanguageServerSuspended);
    };
    // Process any existing documents.
    for (const textDoc of vscode.workspace.textDocuments) {
      await startClientOnOpenDocument(textDoc);
    }

    // Watch any new documents to spawn servers when necessary.
    this.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument(startClientOnOpenDocument));
    this.subscriptions.push(
        vscode.workspace.onDidChangeWorkspaceFolders((event) => {
          for (const folder of event.removed) {
            const client = this.workspaceClients.get(folder.uri.toString());
            if (client) {
              client.stop();
              this.workspaceClients.delete(folder.uri.toString());
            }
          }
        }));

    // Initialize the formatter.
    this.subscriptions.push(registerFormatter(loggingService, this.getSDK()));
    loggingService.logInfo("MojoContext activated.");
  }

  /**
   * Open or return a language server for the given uri and language.
   */
  async getOrActivateLanguageClient(uri: vscode.Uri,
                                    launchLanguageServerSuspended: boolean):
      Promise<vscodelc.LanguageClient|undefined> {
    this.getLoggingService().logInfo(
        `Activating language client for URI '${uri}'`)
    // Check the scheme of the uri.
    let validSchemes = [ 'file' ];
    if (!validSchemes.includes(uri.scheme)) {
      this.getLoggingService().logInfo(`Unsupported URI scheme '${uri.scheme}'`)
      return undefined;
    }

    // Resolve the workspace folder if this document is in one. We use the
    // workspace folder when determining if a server needs to be started.
    let workspaceFolder = vscode.workspace.getWorkspaceFolder(uri);
    let workspaceFolderStr =
        workspaceFolder ? workspaceFolder.uri.toString() : "";

    // Get or create a client context for this folder.
    let client = this.workspaceClients.get(workspaceFolderStr);
    if (!client) {
      client = await this.activateWorkspaceFolder(
          workspaceFolder, this.getLoggingService(),
          launchLanguageServerSuspended);
      if (client) {
        this.workspaceClients.set(workspaceFolderStr, client);
      }
    }
    return client;
  }

  /**
   *  Activate the language client for the given language in the given workspace
   *  folder.
   */
  async activateWorkspaceFolder(workspaceFolder: vscode.WorkspaceFolder|
                                undefined,
                                loggingService: LoggingService,
                                launchLanguageServerSuspended: boolean):
      Promise<vscodelc.LanguageClient|undefined> {
    // Try to activate the language client.
    const [server, serverPath] = await this.startLanguageClient(
        workspaceFolder, loggingService, launchLanguageServerSuspended);

    // Watch for configuration changes on this folder.
    if (workspaceFolder)
      await configWatcher.activate(this, workspaceFolder, [ 'modularHomePath' ],
                                   [ serverPath ]);
    return server;
  }

  /**
   *  Start a new language client. Returns an array containing the opened
   *  server, or null if the server could not be started, and the resolved
   *  server path.
   */
  async startLanguageClient(workspaceFolder: vscode.WorkspaceFolder|undefined,
                            loggingService: LoggingService,
                            launchLanguageServerSuspended: boolean):
      Promise<[ vscodelc.LanguageClient | undefined, string ]> {
    loggingService.logInfo("Starting language client for workspace",
                           workspaceFolder);
    const clientTitle = 'Mojo Language Client';

    // Get the path of the lsp-server that is used to provide language
    // functionality.
    let mojoConfig = await this.getSDK().resolveConfig(workspaceFolder);
    if (!mojoConfig)
      return [ undefined, "" ];

    let args = [];
    if (launchLanguageServerSuspended)
      args.push("--suspended");

    // Configure the server options.
    const serverOptions: vscodelc.ServerOptions = {
      command : mojoConfig.mojoLanguageServerPath,
      args,
      options :
          {env : {...process.env, MODULAR_HOME : mojoConfig.modularHomePath}}
    };

    // This setting is not exposed in package.json because it's internal.
    const env = config.get<{[key: string] : string}>("env", workspaceFolder);
    if (env) {
      for (let [name, value] of Object.entries(env)) {
        // We need to resolve wildcard values manually.
        const resolvedPath = workspaceFolder
                                 ? value.replace("${workspaceFolder}",
                                                 workspaceFolder.uri.fsPath)
                                 : value;
        serverOptions.options!.env[name] = resolvedPath;
      }
    }

    // Configure file patterns relative to the workspace folder.
    let filePattern: vscode.GlobPattern = '**/*.{lit,mojo}';
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
        {language : 'lit', pattern : selectorPattern},
        {language : 'mojo', pattern : selectorPattern},
      ],
      synchronize : {
        // Notify the server about file changes to language files contained in
        // the workspace.
        fileEvents : vscode.workspace.createFileSystemWatcher(filePattern)
      },
      outputChannel : loggingService.outputChannel,
      workspaceFolder : workspaceFolder,
      middleware : middleware,

      // Don't switch to output window when the server returns output.
      revealOutputChannelOn : vscodelc.RevealOutputChannelOn.Never,
    };

    // Create the language client and start the client.
    let languageClient = new vscodelc.LanguageClient(
        'mojo-lsp', clientTitle, serverOptions, clientOptions);
    languageClient.start();
    return [ languageClient, mojoConfig.mojoLanguageServerPath ];
  }

  /**
   * Return the language client for the given language and uri, or null if no
   * client is active.
   */
  getLanguageClient(uri: vscode.Uri): vscodelc.LanguageClient|undefined {
    let workspaceFolder = vscode.workspace.getWorkspaceFolder(uri);
    let workspaceFolderStr =
        workspaceFolder ? workspaceFolder.uri.toString() : "";
    return this.workspaceClients.get(workspaceFolderStr);
  }

  dispose() {
    this.subscriptions.forEach((d) => { d.dispose(); });
    this.subscriptions = [];
    this.workspaceClients.forEach((client) => {
      if (client) {
        client.stop();
      }
    });
    this.workspaceClients.clear();
  }
}
