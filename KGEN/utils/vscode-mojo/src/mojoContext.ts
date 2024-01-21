//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient/node';

import {activateRunCommands} from './commands/run';
import {MojoDebugContext} from './debug/debug';
import {MojoDecoratorContext} from './decorations';
import {registerFormatter} from './formatter';
import {LoggingService} from './logging';
import {MOJOSDK} from './mojoSDK';
import * as config from './utils/config';
import * as configWatcher from './utils/configWatcher';
import {DisposableContext} from './utils/disposableContext';

/**
 *  This class manages the Mojo extension state, including the language
 *  client.
 */
export class MOJOContext extends DisposableContext {
  readonly sdk: MOJOSDK;
  private workspaceClients: Map<string, vscodelc.LanguageClient> = new Map();
  readonly loggingService: LoggingService;
  readonly extensionContext: vscode.ExtensionContext;

  constructor(extensionContext: vscode.ExtensionContext,
              loggingService: LoggingService) {
    super();
    this.extensionContext = extensionContext;
    this.loggingService = loggingService;
    this.sdk = new MOJOSDK(this.loggingService);
  }

  /**
   *  Activate the Mojo context, and start the language clients.
   */
  async activate(launchLanguageServerSuspended: boolean = false) {
    this.loggingService
        .logInfo("Activating the Mojo Context.")

        // Initialize the commands of the extension.
        this.pushSubscription(
            vscode.commands.registerCommand('mojo.restart', async () => {
              // Dispose and reactivate the context.
              this.dispose();
              await this.activate();
            }));
    this.pushSubscription(
        vscode.commands.registerCommand('mojo.restart-suspended', async () => {
          // Dispose and reactivate the context.
          this.dispose();
          await this.activate(/*launchLanguageServerSuspended=*/ true);
        }));

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
    this.pushSubscription(
        vscode.workspace.onDidOpenTextDocument(startClientOnOpenDocument));
    this.pushSubscription(
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
    this.pushSubscription(registerFormatter(this.loggingService, this.sdk));

    // Initialize the debugger support.
    this.pushSubscription(new MojoDebugContext(this));

    // Initialize the execution commands.
    this.pushSubscription(activateRunCommands(this));

    // Initialize the decorations.
    this.pushSubscription(new MojoDecoratorContext());

    this.loggingService.logInfo("MojoContext activated.");
  }

  /**
   * Open or return a language server for the given uri and language.
   */
  async getOrActivateLanguageClient(uri: vscode.Uri,
                                    launchLanguageServerSuspended: boolean):
      Promise<vscodelc.LanguageClient|undefined> {
    if (!uri.fsPath.endsWith(".mojo") && !uri.fsPath.endsWith('🔥') &&
        !uri.fsPath.endsWith(".ipynb"))
      return undefined;

    this.loggingService.logInfo(`Activating language client for URI '${uri}'`)
    // Check the scheme of the uri.
    let validSchemes = [ 'file', 'vscode-notebook-cell' ];
    if (!validSchemes.includes(uri.scheme)) {
      this.loggingService.logInfo(`Unsupported URI scheme '${uri.scheme}'`)
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
          workspaceFolder, this.loggingService, launchLanguageServerSuspended);
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
    let mojoConfig = await this.sdk.resolveConfig(workspaceFolder);
    if (!mojoConfig)
      return [ undefined, "" ];

    let args = [];
    if (launchLanguageServerSuspended)
      args.push("--suspended");

    if (config.get<boolean>("lsp.parseStdlib", workspaceFolder, false))
      args.push("--parse-stdlib");

    // Configure the server options.
    const serverOptions: vscodelc.ServerOptions = {
      command : mojoConfig.mojoLanguageServerPath,
      args,
      options :
          {env : {...process.env, MODULAR_HOME : mojoConfig.modularHomePath}}
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
    this.loggingService.logInfo("Disposing MOJOContext.");
    super.dispose();
    this.workspaceClients.forEach((client) => {
      if (client) {
        client.stop();
      }
    });
    this.workspaceClients.clear();
  }
}
