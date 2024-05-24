//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import {
  DiagnosticSeverity,
  DidChangeTextDocumentParams,
  DidCloseTextDocumentParams,
  DidOpenTextDocumentParams,
  InitializeParams,
  InitializeResult,
  PublishDiagnosticsNotification,
  PublishDiagnosticsParams
} from 'vscode-languageserver-protocol';
import {
  createConnection as createClientConnection,
  ProposedFeatures,
} from 'vscode-languageserver/node';

import {MojoDocument} from './MojoDocument';
import {MojoLSPServer} from './MojoLSPServer';
import {Client, ExitStatus, RequestParamsWithDocument, URI} from './types';

/**
 * Class in charge of of managing the communication between the VSCode client
 * and the actual mojo-lsp-server.
 */
export class MojoLSPProxy {
  /**
   * The connection with the VSCode client.
   */
  private client: Client;
  /**
   * The actual Mojo LSP Server. It'll be created as part of the `onInitialize`
   * method of the proxy.
   */
  private server: MojoLSPServer|undefined;
  /**
   * The list of documents currently tracked by the proxy.
   */
  private trackedDocuments = new Map<URI, MojoDocument>();
  /**
   * A count for how many times the server was restarted.
   */
  private restartCount = 0;
  /**
   * The initialization params used to launch the server. They are gotten from
   * the client as part of the `initialize` request and have to be reused
   * whenever the server is restarted.
   */
  private initializeParams: InitializeParams|undefined;

  constructor() {
    this.client = createClientConnection(ProposedFeatures.all);
    this.registerProxies();
  }

  /**
   * Start the actual communication with the client.
   */
  public start() { this.client.listen(); }

  /**
   * Create a the error message that will be display on the given document upon
   * a crash.
   */
  private createDiagnosticErrorMessageUponCrash(doc: MojoDocument,
                                                crashTrigger: URI|
                                                undefined): string {
    let errorMessage = "A crash happened in the Mojo Language Server";
    if (doc.isCrashTrigger) {
      errorMessage +=
          " when processing this file. The Language Server will try to " +
          "reprocess this file once it is edited again.";
    } else {
      if (crashTrigger !== undefined)
        errorMessage += " when processing " + crashTrigger;
      errorMessage += ". The Language Server will try to reprocess this " +
                      "file automatically.";
    }
    errorMessage +=
        " Please report this issue in " +
        "https://github.com/modularml/mojo/issues along with all the " +
        "relevant source codes with their current contents.";
    return errorMessage;
  }

  /**
   * Whenever there's a restart, this clears the diagnostics for each tracked
   * file and adds one new diagnostic mentioning the crash.
   * We also mark the possible culprit doc appropriately.
   */
  private prepareTrackedDocsForRestart() {
    // In order to identify the crash trigger, we use the simple heuristic of
    // assuming that the oldest pending request is the one that caused the
    // crash. This should work most the times, as most crashes should originate
    // when the server is processing a request. However, if the crash happens at
    // any other moment, e.g., when reading its stdin, we would need a more
    // complex mechanism.
    const crashTrigger =
        (this.server?.getOldestPendingRequest() as RequestParamsWithDocument |
         undefined)
            ?.textDocument?.uri;
    for (const doc of this.trackedDocuments.values()) {
      doc.isCrashTrigger = doc.textDocument.uri === crashTrigger;
      doc.trackedByServer = false;
      const errorMessage =
          this.createDiagnosticErrorMessageUponCrash(doc, crashTrigger);

      const diagnostic: PublishDiagnosticsParams = {
        diagnostics : [ {
          message : errorMessage,
          range : {
            start : {line : 0, character : 0},
            end : {line : 0, character : 0}
          },
          severity : DiagnosticSeverity.Error,
          source : "mojo"
        } ],
        uri : doc.textDocument.uri,
        version : doc.textDocument.version
      };
      this.client.sendNotification(PublishDiagnosticsNotification.method,
                                   diagnostic);
    }
  }

  /**
   * Restart the server upon an unsuccessful termination of the server. This
   * will also issue an initialization request to the new server.
   */
  private restartServer(status: ExitStatus) {
    this.client.console.log(`The mojo-lsp-server binary exited with signal '${
        status.signal}' and exit code '${status.code}'.`);

    this.restartCount++;
    // If we restart too many times, then something weird might be going on,
    // so we just fail the entire proxy for a full restart.
    if (this.restartCount === 100) {
      this.client.console.log(
          "The mojo-lsp-server binary has exited unsuccessfully too many times. The proxy will terminate.");
      if (status.signal !== null)
        process.kill(process.pid, status.signal);
      process.exit(status.code!);
    }
    this.client.console.log(`The mojo-lsp-server will restart.`);
    this.prepareTrackedDocsForRestart();
    this.server!.dispose();
    this.initializeServer();
  }

  /**
   * Spawn a new server and send the initialization request to it.
   *
   * @returns the response to the initialization request.
   */
  private initializeServer(): Promise<InitializeResult> {
    const params = this.initializeParams!;
    const workspaceFolder = params.rootUri;
    this.client.console.log(
        `Server(${process.pid}) ${workspaceFolder} started`);

    this.server = new MojoLSPServer({
      initializationOptions : params.initializationOptions,
      logger : (message: string) => this.client.console.log(message),
      onExit : (status: ExitStatus) => {
        // If the server exited successfully, then that's because a terminate
        // request was sent, so we just terminate the proxy as well.
        if (status.code === 0)
          process.exit(0);
        // There's been an error, we'll try restart the server.
        this.restartServer(status);
      },
      onNotification : (method: string, params: any) =>
          this.client.sendNotification(method, params)
    });
    return this.server!.sendRequest(params, "initialize") as
           Promise<InitializeResult>;
  }

  /**
   * Register the individual proxies for all requests and client-sided
   * notifications supports by the mojo-lsp-server.
   */
  private registerProxies() {
    // Initialize request is special because it contains the information we need
    // to launch the actual mojo-lsp-server.
    this.client.onInitialize(async (params) => {
      this.initializeParams = params;
      return this.initializeServer();
    });

    // Document-based requests
    // Note: all of these requests must go through `relayRequestWithDocument` to
    // ensure crash handling is applied correctly.
    this.client.onCodeAction(
        this.relayRequestWithDocument("textDocument/codeAction"));
    this.client.onCompletion(
        this.relayRequestWithDocument("textDocument/completion"));
    this.client.onDefinition(
        this.relayRequestWithDocument("textDocument/definition"));
    this.client.onDocumentSymbol(
        this.relayRequestWithDocument("textDocument/documentSymbol"));
    this.client.onFoldingRanges(
        this.relayRequestWithDocument("textDocument/foldingRange"));
    this.client.onHover(this.relayRequestWithDocument("textDocument/hover"));
    this.client.onReferences(
        this.relayRequestWithDocument("textDocument/references"));
    this.client.onRenameRequest(
        this.relayRequestWithDocument("textDocument/rename"));
    this.client.onSignatureHelp(
        this.relayRequestWithDocument("textDocument/signatureHelp"));
    this.client.onShutdown((params) => {
      return this.server!.sendRequest(params, "shutdown") as Promise<any>;
    });
    this.client.languages.inlayHint.on(
        this.relayRequestWithDocument("textDocument/inlayHint"));
    this.client.languages.semanticTokens.on(
        this.relayRequestWithDocument("textDocument/semanticTokens/full"));
    this.client.languages.semanticTokens.onDelta(this.relayRequestWithDocument(
        "textDocument/semanticTokens/full/delta"));

    // Client notifications - normal documents
    this.client.onDidOpenTextDocument((params: DidOpenTextDocumentParams) => {
      const doc = new MojoDocument(params.textDocument);
      this.trackedDocuments.set(params.textDocument.uri, doc);
      doc.trackedByServer = true;
      this.server!.sendNotification(params, "textDocument/didOpen");
    });

    this.client.onDidCloseTextDocument((params: DidCloseTextDocumentParams) => {
      this.trackedDocuments.delete(params.textDocument.uri);
      this.server!.sendNotification(params, "textDocument/didClose");
    });

    this.client.onDidChangeTextDocument((params:
                                             DidChangeTextDocumentParams) => {
      const doc = this.trackedDocuments.get(params.textDocument.uri);
      if (!doc) {
        this.client.console.log(
            `Updating a document non-tracked by the proxy '${
                params.textDocument.uri}'.`);
        this.server!.sendNotification(params, "textDocument/didChange");
      } else {
        // If we cannot apply changes locally, we just stop tracking that file,
        // but we still send the notifications as usual to the server just to
        // have additional error logs. This should be an extremely rare error
        // anyway.
        if (!doc.applyChanges(params, this.client)) {
          this.client.console.error(`Couldn't update the document '${
              params.textDocument
                  .uri}' in the proxy. It will stop being tracked by the proxy.`);
          this.trackedDocuments.delete(params.textDocument.uri);
          this.server!.sendNotification(params, "textDocument/didChange");
          return;
        }
        // If the document is not tracked by the server, then we just had a
        // crash. In order to have it tracked by the server, we need to issue a
        // `didOpen` notification with the entire text upon modifications,
        // instead of a `didChange` notification.
        if (!doc.trackedByServer)
          this.openDocManually(doc);
        else
          this.server!.sendNotification(params, "textDocument/didChange");
      }
    });

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
   * Send a manual didOpen notification to the server with the full contents of
   * the doc, as tracked by the proxy.
   */
  private openDocManually(doc: MojoDocument) {
    const didOpenParams: DidOpenTextDocumentParams = {
      textDocument : {
        languageId : doc.textDocument.languageId,
        uri : doc.textDocument.uri,
        text : doc.textDocument.getText(),
        // This version should be different to anything that comes from the IDE.
        version : -1
      }
    };
    doc.trackedByServer = true;
    doc.isCrashTrigger = false;
    this.server!.sendNotification(didOpenParams, "textDocument/didOpen");
  }

  /**
   * This method should be used to relay requests that have a `textDocument.uri`
   * param.
   */
  private relayRequestWithDocument(method: string) {
    return (params: RequestParamsWithDocument) => {
      const uri: URI = params.textDocument.uri;
      const doc = this.trackedDocuments.get(uri);
      // If try to run a request on file that is not tracked by the server,
      // then we need to reopen the doc because we just had a crash recently.
      // However, if it's a crash trigger, we don't reopen it and wait for edits
      // to happen.
      if (doc !== undefined && !doc.isCrashTrigger && !doc.trackedByServer)
        this.openDocManually(doc);
      return this.server!.sendRequest(params, method) as any;
    }
  }

  /**
   * Helper method to reduce boilerplate when setting up a notification proxy.
   */
  private notificationPassthrough(method: string): (params: any) => void {
    return (params: any) => this.server!.sendNotification(params, method);
  }
}
