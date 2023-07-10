import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient/node';

import * as config from './config';
import * as configWatcher from './configWatcher';

/**
 *  This class manages the Mojo extension state, including the language
 *  client.
 */
export class MOJOContext implements vscode.Disposable {
  subscriptions: vscode.Disposable[] = [];
  workspaceClients: Map<string, vscodelc.LanguageClient> = new Map();
  outputChannel: vscode.OutputChannel;
  launchSuspended: boolean;

  /**
   *  Activate the Mojo context, and start the language clients.
   */
  async activate(outputChannel: vscode.OutputChannel, launchSuspended: boolean = false) {
    this.outputChannel = outputChannel;
    this.launchSuspended = launchSuspended;

    // This lambda is used to lazily start language clients for the given
    // document. It removes the need to pro-actively start language clients for
    // every folder within the workspace.
    const startClientOnOpenDocument = async (document: vscode.TextDocument) => {
      await this.getOrActivateLanguageClient(document.uri);
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
  }

  /**
   * Open or return a language server for the given uri and language.
   */
  async getOrActivateLanguageClient(uri: vscode.Uri):
      Promise<vscodelc.LanguageClient> {
    // Check the scheme of the uri.
    let validSchemes = [ 'file' ];
    if (!validSchemes.includes(uri.scheme)) {
      return null;
    }

    // Resolve the workspace folder if this document is in one. We use the
    // workspace folder when determining if a server needs to be started.
    let workspaceFolder = vscode.workspace.getWorkspaceFolder(uri);
    let workspaceFolderStr =
        workspaceFolder ? workspaceFolder.uri.toString() : "";

    // Get or create a client context for this folder.
    let client = this.workspaceClients.get(workspaceFolderStr);
    if (!client) {
      client = await this.activateWorkspaceFolder(workspaceFolder,
                                                  this.outputChannel);
      this.workspaceClients.set(workspaceFolderStr, client);
    }
    return client;
  }

  /**
   *  Activate the language client for the given language in the given workspace
   *  folder.
   */
  async activateWorkspaceFolder(workspaceFolder: vscode.WorkspaceFolder,
                                outputChannel: vscode.OutputChannel):
      Promise<vscodelc.LanguageClient> {
    // Try to activate the language client.
    const [server, serverPath] =
        await this.startLanguageClient(workspaceFolder, outputChannel);

    // Watch for configuration changes on this folder.
    await configWatcher.activate(this, workspaceFolder, [ 'server_path' ],
                                 [ serverPath ]);
    return server;
  }

  /**
   *  Start a new language client. Returns an array containing the opened
   *  server, or null if the server could not be started, and the resolved
   *  server path.
   */
  async startLanguageClient(workspaceFolder: vscode.WorkspaceFolder,
                            outputChannel: vscode.OutputChannel):
      Promise<[ vscodelc.LanguageClient, string ]> {
    const clientTitle = 'Mojo Language Client';

    // Get the path of the lsp-server that is used to provide language
    // functionality.
    var serverPath =
        await this.resolveServerPath('server_path', workspaceFolder);

    // If the server path is empty, bail. We don't emit errors if the user
    // hasn't explicitly configured the server.
    if (serverPath === '') {
      return [ null, serverPath ];
    }

    // Check that the file actually exists.
    if (!fs.existsSync(serverPath)) {
      vscode.window
          .showErrorMessage(
              `${clientTitle}: Unable to resolve path for 'server_path', please ensure the path is correct`,
              "Open Setting")
          .then((value) => {
            if (value === "Open Setting") {
              vscode.commands.executeCommand(
                  'workbench.action.openWorkspaceSettings',
                  {openToSide : false, query : `mojo.server_path`});
            }
          });
      return [ null, serverPath ];
    }

    let args = [];
    if (this.launchSuspended)
      args.push("--suspended");

    // Configure the server options.
    const serverOptions: vscodelc.ServerOptions = {
      command : serverPath,
      args,
    };

    // Configure file patterns relative to the workspace folder.
    let filePattern: vscode.GlobPattern = '**/*.{lit,mojo}';
    let selectorPattern: string = null;
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
        didOpen : (document, next) : Promise<void> => {
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
      outputChannel : outputChannel,
      workspaceFolder : workspaceFolder,
      middleware : middleware,

      // Don't switch to output window when the server returns output.
      revealOutputChannelOn : vscodelc.RevealOutputChannelOn.Never,
    };

    // Create the language client and start the client.
    let languageClient = new vscodelc.LanguageClient(
        'mojo-lsp', clientTitle, serverOptions, clientOptions);
    languageClient.start();
    return [ languageClient, serverPath ];
  }

  /**
   * Try to resolve the given path, or the default path, with an optional
   * workspace folder. If a path could not be resolved, just returns the
   * input filePath.
   */
  async resolvePath(filePath: string, defaultPath: string,
                    workspaceFolder: vscode.WorkspaceFolder): Promise<string> {
    const configPath = filePath;

    // If the path is already fully resolved, there is nothing to do.
    if (path.isAbsolute(filePath)) {
      return filePath;
    }

    // If a path hasn't been set, try to use the default path.
    if (filePath === '') {
      if (defaultPath === '') {
        return filePath;
      }
      filePath = defaultPath;

      // Fallthrough to try resolving the default path.
    }

    // Try to resolve the path relative to the workspace.
    let filePattern: vscode.GlobPattern = '**/' + filePath;
    if (workspaceFolder) {
      filePattern = new vscode.RelativePattern(workspaceFolder, filePattern);
    }
    let foundUris = await vscode.workspace.findFiles(filePattern, null, 1);
    if (foundUris.length === 0) {
      // If we couldn't resolve it, just return the original path anyways. The
      // file might not exist yet.
      return configPath;
    }
    // Otherwise, return the resolved path.
    return foundUris[0].fsPath;
  }

  /**
   * Try to resolve the path for the given server setting, with an optional
   * workspace folder.
   */
  async resolveServerPath(serverSettingName: string,
                          workspaceFolder: vscode.WorkspaceFolder):
      Promise<string> {
    const serverPath = config.get<string>(serverSettingName, workspaceFolder);
    return this.resolvePath(serverPath, 'mojo-lsp-server', workspaceFolder);
  }

  /**
   * Return the language client for the given language and uri, or null if no
   * client is active.
   */
  getLanguageClient(uri: vscode.Uri): vscodelc.LanguageClient {
    let workspaceFolder = vscode.workspace.getWorkspaceFolder(uri);
    let workspaceFolderStr =
        workspaceFolder ? workspaceFolder.uri.toString() : "";
    return this.workspaceClients.get(workspaceFolderStr);
  }

  dispose() {
    this.subscriptions.forEach((d) => { d.dispose(); });
    this.subscriptions = [];
    this.workspaceClients.forEach((client) => { client.stop(); });
    this.workspaceClients.clear();
  }
}
