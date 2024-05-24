//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import {
  createConnection as createClientConnection,
  ProposedFeatures,
} from 'vscode-languageserver/node';

import {MojoLSPServer} from './MojoLSPServer';
import {ExitStatus} from './types';

/**
 * Class in charge of of managing the communication between the VSCode client
 * and the actual mojo-lsp-server.
 */
export class MojoLSPProxy {
  // The connection with the VSCode client.
  private client: ReturnType<typeof createClientConnection>;
  // The actual Mojo LSP Server. It'll be created as part of the `onInitialize`
  // method of the proxy.
  private server: MojoLSPServer|undefined;

  constructor() {
    this.client = createClientConnection(ProposedFeatures.all);
    this.registerProxies();
  }

  /**
   * Start the actual communication with the client.
   */
  public start() { this.client.listen(); }

  /**
   * Register the individual proxies for all requests and client-sided
   * notifications supports by the mojo-lsp-server.
   */
  private registerProxies() {
    // Initialize request is special because it contains the information we need
    // to launch the actual mojo-lsp-server.
    this.client.onInitialize(async (params) => {
      const workspaceFolder = params.rootUri;
      this.client.console.log(`[Server(${process.pid}) ${
          workspaceFolder}] Started and initialize received`);

      this.server = new MojoLSPServer({
        initializationOptions : params.initializationOptions,
        logger : (message: string) => this.client.console.log(message),
        onExit : (status: ExitStatus) => {
          if (status.signal !== undefined)
            process.kill(process.pid, status.signal as string);
          process.exit(status.code!);
        },
        onNotification : (method: string, params: any) =>
            this.client.sendNotification(method, params)
      });

      return this.server.sendRequest(params, "initialize");
    });

    // Requests
    this.client.onCodeAction(
        this.requestPassthrough("textDocument/codeAction"));
    this.client.onCompletion(
        this.requestPassthrough("textDocument/completion"));
    this.client.onDefinition(
        this.requestPassthrough("textDocument/definition"));
    this.client.onDocumentSymbol(
        this.requestPassthrough("textDocument/documentSymbol"));
    this.client.onFoldingRanges(
        this.requestPassthrough("textDocument/foldingRange"));
    this.client.onHover(this.requestPassthrough("textDocument/hover"));
    this.client.onRenameRequest(this.requestPassthrough("textDocument/rename"));
    this.client.onReferences(
        this.requestPassthrough("textDocument/references"));
    this.client.onSignatureHelp(
        this.requestPassthrough("textDocument/signatureHelp"));
    this.client.onShutdown(this.requestPassthrough("shutdown"));
    this.client.languages.inlayHint.on(
        this.requestPassthrough("textDocument/inlayHint"));
    this.client.languages.semanticTokens.on(
        this.requestPassthrough("textDocument/semanticTokens/full"));
    this.client.languages.semanticTokens.onDelta(
        this.requestPassthrough("textDocument/semanticTokens/full/delta"));

    // Client notifications - normal documents
    this.client.onDidOpenTextDocument(
        this.notificationPassthrough("textDocument/didOpen"));
    this.client.onDidCloseTextDocument(
        this.notificationPassthrough("textDocument/didClose"));
    this.client.onDidChangeTextDocument(
        this.notificationPassthrough("textDocument/didChange"));

    // Client notifications - notebooks
    const notebooks = this.client.notebooks.synchronization;
    notebooks.onDidOpenNotebookDocument(
        this.notificationPassthrough("notebookDocument/didOpen"));
    notebooks.onDidCloseNotebookDocument(
        this.notificationPassthrough("notebookDocument/didClose"));
    notebooks.onDidChangeNotebookDocument(
        this.notificationPassthrough("notebookDocument/didChange"));
  }

  /**
   * Helper method to reduce boilerplate when setting up a request proxy.
   */
  private requestPassthrough(method: string): (params: any) => Promise<any> {
    return (params: any) => this.server!.sendRequest(params, method);
  }

  /**
   * Helper method to reduce boilerplate when setting up a notification proxy.
   */
  private notificationPassthrough(method: string): (params: any) => void {
    return (params: any) => this.server!.sendNotification(params, method);
  }
}
