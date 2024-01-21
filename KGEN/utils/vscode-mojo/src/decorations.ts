//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import {MOJOContext} from './mojoContext';
import {DisposableContext} from './utils/disposableContext';

/**
 * MojoDecoratorContext is responsible for decorating mojo documents with
 * additional information.
 */
export class MojoDecoratorContext extends DisposableContext {
  private docStringDecorationType: vscode.TextEditorDecorationType;

  constructor() {
    super();

    // Create a decoration type for doc strings. The decoration adds a `>`
    // before the doc string, to help visually distinguish it from the rest of
    // the code. This effectively models an inlay hint, but we use a decoration
    // type as decorations get refreshed much faster.
    this.docStringDecorationType =
        vscode.window.createTextEditorDecorationType({
          after : {
            contentText : '>',
            color : {id : 'editorInlayHint.foreground'},
            backgroundColor : {id : 'editorInlayHint.background'},

            // Add a little padding to the right of the inlay hint.
            margin : '0em 0.2em 0em 0em',
          },
          // Hide the decoration, we only care about the "after" content.
          opacity : "0"
        });

    this.pushSubscription(vscode.workspace.onDidOpenTextDocument(
        event => { this.decorateDocument(event); }));
    this.pushSubscription(vscode.workspace.onDidChangeTextDocument(
        event => { this.decorateDocument(event.document); }));
    this.pushSubscription(vscode.workspace.onDidOpenNotebookDocument(
        event => { this.decorateNotebookDocument(event); }));
    this.pushSubscription(vscode.workspace.onDidChangeNotebookDocument(
        event => { this.decorateNotebookDocument(event.notebook); }));
    this.pushSubscription(vscode.window.onDidChangeVisibleTextEditors(
        editors => { editors.forEach(editor => this.decorate(editor)); }));
    this.pushSubscription(
        vscode.window.onDidChangeVisibleNotebookEditors(editors => {
          editors.forEach(editor =>
                              this.decorateNotebookDocument(editor.notebook));
        }));

    // Process any existing documents.
    for (const textDoc of vscode.workspace.textDocuments)
      this.decorateDocument(textDoc);
  }

  private decorateDocument(document: vscode.TextDocument) {
    // Check if the document is a mojo document.
    if (document.languageId !== 'mojo')
      return;
    // Check if this is one of the visible editors.
    vscode.window.visibleTextEditors.forEach(editor => {
      if (editor.document === document)
        this.decorate(editor);
    });
  }

  private decorateNotebookDocument(notebook: vscode.NotebookDocument) {
    vscode.window.visibleNotebookEditors.forEach(editor => {
      if (editor.notebook !== notebook)
        return;

      // Decorate any mojo cells in the notebook.
      for (let cell of notebook.getCells())
        this.decorateDocument(cell.document)
    });
  }

  /**
   * Generate decorations for the given text document. This includes decorations
   * for doc string code blocks, etc.
   */
  private decorate(editor: vscode.TextEditor) {
    const text = editor.document.getText();
    const splitLines = text.split('\n');
    let docDectorations: vscode.DecorationOptions[] = [];

    // Generate decorations for code blocks in the document. This helps
    // visually distinguish code blocks from the rest of the document.
    const startRegEx = /^( *)\`{3,}mojo$/g;
    const endRegEx = /^ *\`{3,}$/g;
    let numCurrentCodeBlocks = 0;
    let currentCodeBlockIndent = 0;
    let prevNumDecorations = 0;
    for (let line = 0, lineE = splitLines.length; line != lineE; ++line) {
      // Check for the start of a new codeblock.
      let match;
      if ((match = startRegEx.exec(splitLines[line]))) {
        if (numCurrentCodeBlocks++ == 0) {
          currentCodeBlockIndent = match[1].length;
          prevNumDecorations = docDectorations.length;
        }
        continue;
      }
      if (numCurrentCodeBlocks) {
        // Check for the end of a codeblock.
        if (endRegEx.test(splitLines[line])) {
          --numCurrentCodeBlocks;
          continue;
        }
        let lineLength = splitLines[line].length;
        if (lineLength <= currentCodeBlockIndent)
          continue;

        // Add a decoration for this code block.
        let pos = new vscode.Position(line, currentCodeBlockIndent)
        docDectorations.push({range : new vscode.Range(pos, pos)});
      }
    }

    // If we have a partial code block, remove the decorations.
    if (numCurrentCodeBlocks)
      docDectorations.splice(prevNumDecorations);

    editor.setDecorations(this.docStringDecorationType, docDectorations);
  }
}
