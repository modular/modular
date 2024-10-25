import { NotebookCell } from 'vscode-languageserver';
import {
  DidChangeNotebookDocumentParams,
  DidChangeTextDocumentParams,
  DidCloseNotebookDocumentParams,
  DidCloseTextDocumentParams,
  DidOpenNotebookDocumentParams,
  DidOpenTextDocumentParams,
  NotebookDocument,
  TextDocumentItem,
} from 'vscode-languageserver-protocol';
import { TextDocument } from 'vscode-languageserver-textdocument';

import { MojoLSPServer } from './MojoLSPServer';
import { Client, URI, Optional } from './types';

/**
 * Base class for all kinds of Mojo documents.
 */
export abstract class MojoDocument {
  uri: URI;
  version: number;

  constructor(uri: URI, version: number) {
    this.uri = uri;
    this.version = version;
  }

  /**
   * Send a manual didOpen notification to the server with the full contents of
   * the doc, as tracked by the proxy.
   */
  abstract openDocumentOnServer(
    server: MojoLSPServer,
    stateHandler: MojoDocumentsStateHandler,
  ): void;
}

/**
 * Class that represents a regular text document tracked by the proxy.
 */
export class MojoTextDocument extends MojoDocument {
  /**
   * The underlying raw text document metadata.
   */
  textDocument: TextDocument;

  constructor(params: TextDocumentItem) {
    super(params.uri, params.version);
    this.textDocument = TextDocument.create(
      params.uri,
      params.languageId,
      params.version,
      params.text,
    );
  }

  openDocumentOnServer(
    server: MojoLSPServer,
    stateHandler: MojoDocumentsStateHandler,
  ): void {
    const didOpenParams: DidOpenTextDocumentParams = {
      textDocument: {
        languageId: this.textDocument.languageId,
        uri: this.textDocument.uri,
        text: this.textDocument.getText(),
        version: this.textDocument.version,
      },
    };
    stateHandler.markDocAsTrackedByServer(this);
    stateHandler.crashTriggerUris.delete(this.uri);
    server.sendNotification(didOpenParams, 'textDocument/didOpen');
  }
}

/**
 * Class that represents a notebook cell document tracked by the proxy.
 */
export class MojoNotebookCellDocument extends MojoTextDocument {
  /**
   * The notebook that owns this cell.
   */
  public parentNotebook: MojoNotebookDocument;
  /**
   * The underlying raw cell metadata.
   */
  public notebookCell: NotebookCell;

  constructor(
    params: TextDocumentItem,
    notebookCell: NotebookCell,
    parentNotebook: MojoNotebookDocument,
  ) {
    super(params);
    this.notebookCell = notebookCell;
    this.parentNotebook = parentNotebook;
  }
}

/**
 * Class that represents a notebook document tracked by the proxy.
 */
export class MojoNotebookDocument extends MojoDocument {
  /**
   * The list of cell documents that make this notebook.
   */
  public cellDocs: MojoNotebookCellDocument[] = [];
  /**
   * The underlying notebook metadata.
   */
  public notebookDocument: NotebookDocument;

  constructor(params: DidOpenNotebookDocumentParams) {
    super(params.notebookDocument.uri, params.notebookDocument.version);
    this.notebookDocument = params.notebookDocument;
    this.cellDocs = params.cellTextDocuments.map(
      (item, index) =>
        new MojoNotebookCellDocument(
          item,
          params.notebookDocument.cells[index],
          this,
        ),
    );
  }

  openDocumentOnServer(
    server: MojoLSPServer,
    stateHandler: MojoDocumentsStateHandler,
  ): void {
    const didOpenParams: DidOpenNotebookDocumentParams = {
      notebookDocument: {
        cells: this.cellDocs.map((cellDoc) => cellDoc.notebookCell),
        notebookType: this.notebookDocument.notebookType,
        uri: this.notebookDocument.uri,
        version: this.notebookDocument.version,
      },
      cellTextDocuments: this.cellDocs.map((cellDoc) => ({
        languageId: cellDoc.textDocument.languageId,
        text: cellDoc.textDocument.getText(),
        uri: cellDoc.textDocument.uri,
        version: -1,
      })),
    };
    stateHandler.markDocAsTrackedByServer(this);
    stateHandler.crashTriggerUris.delete(this.uri);
    server.sendNotification(didOpenParams, 'notebookDocument/didOpen');
  }
}

/**
 * Class that is in charge of handling and tracking changes on documents,
 * including their notifications and additional bits like crash information.
 */
export class MojoDocumentsStateHandler {
  /**
   * Map of all the notebook the proxy is aware of.
   */
  public uriToNotebookDocs = new Map<URI, MojoNotebookDocument>();
  /**
   * Map of all the cells the proxy is aware of.
   */
  public uriToCellDocs = new Map<URI, MojoNotebookCellDocument>();
  /**
   * Map of all the text docs the proxy is aware of.
   */
  public uriToTextDocs = new Map<URI, MojoTextDocument>();
  /**
   * Set of all the documents tracked by the server given by their URIs.
   */
  public urisTrackedByServer = new Set<URI>();
  /**
   * Set of all the documents that triggered a crash given by their URIs.
   * If a notebook cell has caused a crash, then it's expected that the parent
   * notebook is also marked as a crash trigger.
   */
  public crashTriggerUris = new Set<URI>();
  /**
   * The LSP client.
   */
  private client: Client;

  constructor(client: Client) {
    this.client = client;
  }

  /**
   * Mark a document as being tracked by the server, including its cells.
   */
  public markDocAsTrackedByServer(doc: MojoDocument) {
    this.urisTrackedByServer.add(doc.uri);

    if (doc instanceof MojoNotebookDocument) {
      for (const cellDoc of doc.cellDocs) {
        this.markDocAsTrackedByServer(cellDoc);
      }
    }
  }

  /**
   * Mark a document as being untracked by the server, including its cells.
   */
  public markDocAsUntrackedByServer(doc: MojoDocument) {
    this.urisTrackedByServer.delete(doc.uri);

    if (doc instanceof MojoNotebookDocument) {
      for (const cellDoc of doc.cellDocs) {
        this.markDocAsUntrackedByServer(cellDoc);
      }
    }
  }

  /**
   * @returns whether the given document is being tracked by the server.
   */
  public isTrackedByServer(doc: MojoDocument): boolean {
    return this.urisTrackedByServer.has(doc.uri);
  }

  /**
   * @returns whether the given document has caused a crash in its current
   *     version.
   */
  public isCrashTrigger(doc: MojoDocument): boolean {
    return this.crashTriggerUris.has(doc.uri);
  }

  /**
   * Marks the given document as a crash trigger in its current version.
   *
   * If it's a notebook cell, then the parent notebook is also marked as a crash
   * trigger.
   */
  public markAsCrashTrigger(doc: MojoDocument): void {
    this.crashTriggerUris.add(doc.uri);

    if (doc instanceof MojoNotebookCellDocument) {
      this.crashTriggerUris.add(doc.parentNotebook.uri);
    }
  }

  /**
   * Update the underlying `cells` based on the incoming list of changes.
   *
   * @returns whether the changes could effectively be applied or not.
   */
  private applyChangesToNotebookDoc(
    changes: DidChangeNotebookDocumentParams,
    doc: MojoNotebookDocument,
  ) {
    const version = changes.notebookDocument.version;
    doc.version = version;

    const cells = changes.change.cells;
    if (cells !== undefined) {
      const array = cells.structure?.array;
      if (array !== undefined) {
        array.cells = array.cells || [];

        const newCellDocs = array.cells.map(
          (cell) =>
            new MojoNotebookCellDocument(
              {
                languageId: 'mojo',
                uri: cell.document,
                version: version,
                text: '',
              },
              cell,
              doc,
            ),
        );

        for (let i = array.start; i < array.start + array.deleteCount; i++) {
          this.uriToCellDocs.delete(doc.cellDocs[i].uri);
        }

        for (const newCellDoc of newCellDocs) {
          this.uriToCellDocs.set(newCellDoc.uri, newCellDoc);
        }

        doc.cellDocs.splice(array.start, array.deleteCount, ...newCellDocs);
        doc.notebookDocument.cells.splice(
          array.start,
          array.deleteCount,
          ...array.cells,
        );
      }
      for (const cellData of cells.data || []) {
        const cellDoc = this.uriToCellDocs.get(cellData.document);

        if (cellDoc !== undefined) {
          cellDoc.notebookCell.kind = cellData.kind;
        }
      }
      for (const textContent of cells.textContent || []) {
        const cellDoc = this.uriToCellDocs.get(textContent.document.uri);
        if (cellDoc !== undefined) {
          if (
            !this.applyChangesToTextDocument(
              {
                contentChanges: textContent.changes,
                textDocument: textContent.document,
              },
              cellDoc,
            )
          ) {
            return false;
          }
        }
      }
    }
    return true;
  }

  /**
   * Generic dispatcher for applying changes to a doc.
   */
  private applyChangesToDoc(params: any, doc: MojoDocument): boolean {
    if (doc instanceof MojoNotebookDocument) {
      return this.applyChangesToNotebookDoc(params, doc);
    }

    if (doc instanceof MojoTextDocument) {
      return this.applyChangesToTextDocument(params, doc);
    }
    throw new Error('unreachable');
  }

  /**
   * Stop tracking a document in the proxy.
   */
  private stopTrackingDocument(doc: MojoDocument) {
    if (doc instanceof MojoNotebookDocument) {
      this.uriToNotebookDocs.delete(doc.uri);
      this.crashTriggerUris.delete(doc.uri);
      this.urisTrackedByServer.delete(doc.uri);

      for (const cellDoc of doc.cellDocs) {
        this.stopTrackingDocument(cellDoc);
      }
    } else if (doc instanceof MojoTextDocument) {
      this.uriToTextDocs.delete(doc.uri);
    } else {
      throw new Error('unreachable');
    }
  }

  /**
   * Generic document change handler.
   */
  public onDidChangeDocument(
    params: any,
    originalNotification: string,
    server: MojoLSPServer,
    uri: URI,
    doc: Optional<MojoDocument>,
  ): void {
    if (!doc) {
      this.client.console.log(
        `Updating a document non-tracked by the proxy '${uri}'.`,
      );
      server.sendNotification(params, originalNotification);
      return;
    }

    // If we cannot apply changes locally, we just stop tracking that file,
    // but we still send the notifications as usual to the server just to
    // have additional error logs. This should be an extremely rare error
    // anyway.
    if (!this.applyChangesToDoc(params, doc)) {
      this.client.console.error(
        `Couldn't update the document '${
          params.notebookDocument.uri
        }' in the proxy. It will stop being tracked by the proxy.`,
      );
      this.stopTrackingDocument(doc);

      server.sendNotification(params, originalNotification);
      return;
    }
    // If the document is not tracked by the server, then we just had a
    // crash. In order to have it tracked by the server, we need to issue a
    // `didOpen` notification with the entire text upon modifications,
    // instead of a `didChange` notification.

    // If the document is not tracked by the server, then we just had a
    // crash. In order to have it tracked by the server, we need to issue a
    // `didOpen` notification with the entire text upon modifications,
    // instead of a `didChange` notification.
    if (!this.isTrackedByServer(doc)) {
      doc.openDocumentOnServer(server, this);
    } else {
      server.sendNotification(params, originalNotification);
    }
  }

  /**
   * "notebookDocument/didChange" handler.
   */
  public onDidChangeNotebookDocument(
    params: DidChangeNotebookDocumentParams,
    server: MojoLSPServer,
  ): void {
    const doc = this.uriToNotebookDocs.get(params.notebookDocument.uri);
    this.onDidChangeDocument(
      params,
      'notebookDocument/didChange',
      server,
      params.notebookDocument.uri,
      doc,
    );
  }

  /**
   * "notebookDocument/didOpen" handler.
   */
  public onDidOpenNotebookDocument(
    params: DidOpenNotebookDocumentParams,
    server: MojoLSPServer,
  ) {
    const doc = new MojoNotebookDocument(params);
    doc.openDocumentOnServer(server, this);
    this.uriToNotebookDocs.set(params.notebookDocument.uri, doc);

    for (const cellDoc of doc.cellDocs) {
      this.uriToCellDocs.set(cellDoc.uri, cellDoc);
    }
  }

  /**
   * "notebookDocument/didClose" handler.
   */
  public onDidCloseNotebookDocument(
    params: DidCloseNotebookDocumentParams,
    server: MojoLSPServer,
  ) {
    const doc = this.uriToNotebookDocs.get(params.notebookDocument.uri);

    if (doc !== undefined) {
      this.stopTrackingDocument(doc);
    }
    this.uriToNotebookDocs.delete(params.notebookDocument.uri);
    server.sendNotification(params, 'notebookDocument/didClose');
  }

  /**
   * Update the underlying `textDocument` based on the incoming list of changes.
   *
   * @returns whether the changes could effectively be applied or not.
   */
  public applyChangesToTextDocument(
    changes: DidChangeTextDocumentParams,
    doc: MojoTextDocument,
  ): boolean {
    try {
      TextDocument.update(
        doc.textDocument,
        changes.contentChanges,
        changes.textDocument.version,
      );
      return true;
    } catch (ex) {
      this.client.console.error(`${ex}`);
      return false;
    }
  }

  /**
   * "textDocument/didChange" handler.
   */
  public onDidChangeTextDocument(
    params: DidChangeTextDocumentParams,
    server: MojoLSPServer,
  ): void {
    const doc = this.uriToTextDocs.get(params.textDocument.uri);
    this.onDidChangeDocument(
      params,
      'textDocument/didChange',
      server,
      params.textDocument.uri,
      doc,
    );
  }

  /**
   * "textDocument/didOpen" handler.
   */
  public onDidOpenTextDocument(
    params: DidOpenTextDocumentParams,
    server: MojoLSPServer,
  ) {
    const doc = new MojoTextDocument(params.textDocument);
    doc.openDocumentOnServer(server, this);
    this.uriToTextDocs.set(doc.uri, doc);
  }

  /**
   * "textDocument/didClose" handler.
   */
  public onDidCloseTextDocument(
    params: DidCloseTextDocumentParams,
    server: MojoLSPServer,
  ) {
    const doc = this.uriToTextDocs.get(params.textDocument.uri);

    if (doc !== undefined) {
      this.stopTrackingDocument(doc);
    }
    server.sendNotification(params, 'textDocument/didClose');
  }

  /**
   * Generator for all tracked docs, including cells.
   */
  public *getAllDocs() {
    for (const map of [
      this.uriToCellDocs,
      this.uriToNotebookDocs,
      this.uriToTextDocs,
    ]) {
      for (const doc of map.values()) {
        yield doc;
      }
    }
  }

  /**
   * @returns the owning notebook if `uri` corresponds to a cell. Otherwise,
   *     returns the document given by its `uri`.
   */
  public getOwningTextOrNotebookDocument(
    uri: URI,
  ): Optional<MojoNotebookDocument | MojoTextDocument> {
    {
      const doc = this.uriToTextDocs.get(uri);

      if (doc !== undefined) {
        return doc;
      }
    }
    {
      const doc = this.uriToNotebookDocs.get(uri);

      if (doc !== undefined) {
        return doc;
      }
    }
    {
      const doc = this.uriToCellDocs.get(uri);

      if (doc !== undefined) {
        return doc.parentNotebook;
      }
    }
    return undefined;
  }
}
