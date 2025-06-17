//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import {
  DiagnosticSeverity,
  DidChangeNotebookDocumentParams,
  DidChangeTextDocumentParams,
  DidCloseNotebookDocumentParams,
  DidCloseTextDocumentParams,
  DidOpenNotebookDocumentParams,
  DidOpenTextDocumentParams,
  InitializeParams,
  InitializeResult,
  PublishDiagnosticsNotification,
  PublishDiagnosticsParams,
} from 'vscode-languageserver-protocol';
import {
  createConnection as createClientConnection,
  ProposedFeatures,
} from 'vscode-languageserver/node';

import { MojoDocument, MojoDocumentsStateHandler } from './MojoDocument';
import { MojoLSPServer } from './MojoLSPServer';
import {
  Client,
  ExitStatus,
  JSONObject,
  Optional,
  RequestParamsWithDocument,
  URI,
} from './types';

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
  private server: Optional<MojoLSPServer>;
  /**
   * The state handler for all the documents notified by the client.
   */
  private docsStateHandler: MojoDocumentsStateHandler;
  /**
   * The time when the proxy was initialized.
   */
  private initTime = Date.now();
  /**
   * The initialization params used to launch the server. They are gotten from
   * the client as part of the `initialize` request and have to be reused
   * whenever the server is restarted.
   */
  private initializeParams: Optional<InitializeParams>;

  constructor() {
    this.client = createClientConnection(ProposedFeatures.all);
    this.docsStateHandler = new MojoDocumentsStateHandler(this.client);
    this.registerProxies();
  }

  /**
   * Start the actual communication with the client.
   */
  public start() {
    this.client.listen();
  }

  /**
   * Create a the error message that will be display on the given document upon
   * a crash.
   */
  private createDiagnosticErrorMessageUponCrash(
    doc: MojoDocument,
    crashTrigger: Optional<URI>,
  ): string {
    let errorMessage = 'A crash happened in the Mojo Language Server';
    if (this.docsStateHandler.isCrashTrigger(doc)) {
      errorMessage +=
        ' when processing this document. The Language Server will try to ' +
        'reprocess this document once it is edited again.';
    } else {
      if (crashTrigger !== undefined) {
        errorMessage += ' when processing ' + crashTrigger;
      }
      errorMessage +=
        '. The Language Server will try to reprocess this ' +
        'document automatically.';
    }
    errorMessage +=
      ' Please report this issue in ' +
      'https://github.com/modular/modular/issues along with all the ' +
      'relevant source codes with their current contents.';
    return errorMessage;
  }

  /**
   * Whenever there's a restart, this clears the diagnostics for each tracked
   * file and adds one new diagnostic mentioning the crash.
   * We also mark the possible culprit doc appropriately.
   */
  private prepareTrackedDocsForRestart() {
    this.docsStateHandler.urisTrackedByServer.clear();
    // In order to identify the crash trigger, we use the simple heuristic of
    // assuming that the oldest pending request is the one that caused the
    // crash. This should work most the times, as most crashes should originate
    // when the server is processing a request. However, if the crash happens at
    // any other moment, e.g., when reading its stdin, we would need a more
    // complex mechanism to identify the actual issue.
    const crashTriggerURI = (
      this.server?.getOldestPendingRequest() as Optional<RequestParamsWithDocument>
    )?.textDocument?.uri;
    for (const doc of this.docsStateHandler.getAllDocs()) {
      if (doc.uri === crashTriggerURI) {
        this.docsStateHandler.markAsCrashTrigger(doc);
      }
      const errorMessage = this.createDiagnosticErrorMessageUponCrash(
        doc,
        crashTriggerURI,
      );

      const diagnostic: PublishDiagnosticsParams = {
        diagnostics: [
          {
            message: errorMessage,
            range: {
              start: { line: 0, character: 0 },
              end: { line: 0, character: 0 },
            },
            severity: DiagnosticSeverity.Error,
            source: 'mojo',
          },
        ],
        uri: doc.uri,
        version: doc.version,
      };
      this.client.sendNotification(
        PublishDiagnosticsNotification.method,
        diagnostic,
      );
    }
  }

  /**
   * Restart the server upon an unsuccessful termination of the server. This
   * will also issue an initialization request to the new server.
   */
  private restartServer(status: ExitStatus) {
    this.client.console.log(
      `The mojo-lsp-server binary exited with signal '${
        status.signal
      }' and exit code '${status.code}'.`,
    );

    this.client.sendNotification('mojo/lspRestart');

    const timeSinceInitInMillis = Date.now() - this.initTime;
    const timeSinceInitInMins = Math.floor(timeSinceInitInMillis / 60000);
    // We only allow one restart per minute to prevent VSCode from disabling the
    // LSP. VSCode allows 4 crashes every 3 minutes.
    if (timeSinceInitInMins >= 1) {
      this.client.console.log(
        `The mojo-lsp-server binary has exited unsuccessfully. The proxy will terminate. It ran ${
          timeSinceInitInMins
        } ms.`,
      );

      if (status.signal !== null) {
        process.kill(process.pid, status.signal);
      }
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
      `Server(${process.pid}) ${workspaceFolder} started`,
    );

    this.server = new MojoLSPServer({
      initializationOptions: params.initializationOptions,
      logger: (message: string) => this.client.console.log(message),
      onExit: (status: ExitStatus) => {
        // If the server exited successfully, then that's because a terminate
        // request was sent, so we just terminate the proxy as well.

        // If the server exited successfully, then that's because a terminate
        // request was sent, so we just terminate the proxy as well.
        if (status.code === 0) {
          process.exit(0);
        }
        // There's been an error, we'll try restart the server.
        // There's been an error, we'll try restart the server.
        this.restartServer(status);
      },
      onNotification: (method: string, params: any) =>
        this.client.sendNotification(method, params),
      onOutgoingRequest: async (
        id: any,
        method: string,
        params: JSONObject,
      ) => {
        let result = await this.client.sendRequest(method, params);
        this.server!.sendResponse(id, result);
      },
    });
    return this.server!.sendRequest(
      params,
      'initialize',
    ) as Promise<InitializeResult>;
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
      this.relayRequestWithDocument('textDocument/codeAction'),
    );
    this.client.onCompletion(
      this.relayRequestWithDocument('textDocument/completion'),
    );
    this.client.onDefinition(
      this.relayRequestWithDocument('textDocument/definition'),
    );
    this.client.onDocumentSymbol(
      this.relayRequestWithDocument('textDocument/documentSymbol'),
    );
    this.client.onFoldingRanges(
      this.relayRequestWithDocument('textDocument/foldingRange'),
    );
    this.client.onHover(this.relayRequestWithDocument('textDocument/hover'));
    this.client.onReferences(
      this.relayRequestWithDocument('textDocument/references'),
    );
    this.client.onRenameRequest(
      this.relayRequestWithDocument('textDocument/rename'),
    );
    this.client.onSignatureHelp(
      this.relayRequestWithDocument('textDocument/signatureHelp'),
    );
    this.client.onShutdown((params) => {
      return this.server!.sendRequest(params, 'shutdown') as Promise<any>;
    });
    this.client.languages.inlayHint.on(
      this.relayRequestWithDocument('textDocument/inlayHint'),
    );
    this.client.languages.semanticTokens.on(
      this.relayRequestWithDocument('textDocument/semanticTokens/full'),
    );
    this.client.languages.semanticTokens.onDelta(
      this.relayRequestWithDocument('textDocument/semanticTokens/full/delta'),
    );

    // Client notifications - normal documents
    this.client.onDidOpenTextDocument((params: DidOpenTextDocumentParams) => {
      this.docsStateHandler.onDidOpenTextDocument(params, this.server!);
    });

    this.client.onDidCloseTextDocument((params: DidCloseTextDocumentParams) => {
      this.docsStateHandler.onDidCloseTextDocument(params, this.server!);
    });

    this.client.onDidChangeTextDocument(
      (params: DidChangeTextDocumentParams) => {
        this.docsStateHandler.onDidChangeTextDocument(params, this.server!);
      },
    );

    // Client notifications - notebooks
    const notebooks = this.client.notebooks.synchronization;
    notebooks.onDidOpenNotebookDocument(
      (params: DidOpenNotebookDocumentParams) => {
        this.docsStateHandler.onDidOpenNotebookDocument(params, this.server!);
      },
    );

    notebooks.onDidCloseNotebookDocument(
      (params: DidCloseNotebookDocumentParams) => {
        this.docsStateHandler.onDidCloseNotebookDocument(params, this.server!);
      },
    );

    notebooks.onDidChangeNotebookDocument(
      (params: DidChangeNotebookDocumentParams) => {
        this.docsStateHandler.onDidChangeNotebookDocument(params, this.server!);
      },
    );

    this.client.onNotification('mojo/emitParsedIR', (params) => {
      this.client.console.log(JSON.stringify(params));
      this.server!.sendNotification(params, 'mojo/emitParsedIR');
    });
  }

  /**
   * This method should be used to relay requests that have a `textDocument.uri`
   * param.
   */
  private relayRequestWithDocument(method: string) {
    return (params: RequestParamsWithDocument) => {
      const uri: URI = params.textDocument.uri;
      // If try to run a request on a document that is not tracked by the
      // server, then we need to reopen it because we just had a crash recently.
      // However, if it's a crash trigger, we don't reopen it and wait for edits
      // to happen first.
      const owningDoc =
        this.docsStateHandler.getOwningTextOrNotebookDocument(uri);

      if (
        owningDoc !== undefined &&
        !this.docsStateHandler.isCrashTrigger(owningDoc) &&
        !this.docsStateHandler.isTrackedByServer(owningDoc)
      ) {
        owningDoc.openDocumentOnServer(this.server!, this.docsStateHandler);
      }
      return this.server!.sendRequest(params, method) as any;
    };
  }
}
