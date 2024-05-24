import {
  DidChangeTextDocumentParams,
  TextDocumentItem
} from "vscode-languageserver-protocol";
import {TextDocument} from 'vscode-languageserver-textdocument';

import {Client} from "./types";

/**
 * Class that represents a Document tracked by the proxy.
 */
export class MojoDocument {
  /**
   * The underlying raw text document metadata.
   */
  textDocument: TextDocument;
  /**
   * Whether the current instance of server knows about this document.
   */
  trackedByServer: boolean = false;
  /**
   * Whether this document, with its current contents, caused a crash.
   */
  isCrashTrigger: boolean = false;

  constructor(params: TextDocumentItem) {
    this.textDocument = TextDocument.create(params.uri, params.languageId,
                                            params.version, params.text);
  }

  /**
   * Update the underlying `textDocument` based on the incoming list of changes.
   *
   * @returns whether the changes could effectively be applied or not.
   */
  public applyChanges(changes: DidChangeTextDocumentParams,
                      client: Client): boolean {
    try {
      TextDocument.update(this.textDocument, changes.contentChanges,
                          changes.textDocument.version);
      return true;
    } catch (ex) {
      client.console.error(`${ex}`);
      return false;
    }
  }
}
