//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains structs for LSP commands that are specific to the Mojo
// server, or structures that have yet to be upstreamed to MLIR.
//
// Each struct has a toJSON and fromJSON function, that converts between
// the struct and a JSON representation. (See JSON.h)
//
// Some structs also have operator<< serialization. This is for debugging and
// tests, and is not generally machine-readable.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_LSP_SERVER_PROTOCOL_H
#define KGEN_TOOLS_MOJO_LSP_SERVER_PROTOCOL_H

#include "mlir/Tools/lsp-server-support/Protocol.h"

namespace mlir {
namespace lsp {
using NotebookDocumentIdentifier = TextDocumentIdentifier;
using VersionedNotebookDocumentIdentifier = VersionedTextDocumentIdentifier;

//===----------------------------------------------------------------------===//
// NotebookCell
//===----------------------------------------------------------------------===//

enum class NotebookCellKind {
  /// A markup-cell is formatted source that is used for display.
  Markup = 1,

  /// A code-cell is source code.
  Code = 2,
};

/// A notebook cell.
///
/// A cell's document URI must be unique across ALL notebook cells and can
/// therefore be used to uniquely identify a notebook cell or the cell's text
/// document.
struct NotebookCell {
  /// The cell's kind.
  NotebookCellKind kind;

  /// The URI of the cell's text document content.
  URIForFile document;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, NotebookCell &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// NotebookDocument
//===----------------------------------------------------------------------===//

struct NotebookDocument {
  /// The notebook document's URI.
  URIForFile uri;

  /// The type of the notebook.
  std::string notebookType;

  /// The version number of this document (it will increase after each change,
  /// including undo/redo).
  int version;

  /// The cells of a notebook.
  std::vector<NotebookCell> cells;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, NotebookDocument &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// DidOpenNotebookDocumentParams
//===----------------------------------------------------------------------===//

struct DidOpenNotebookDocumentParams {
  /// The notebook document that got opened.
  NotebookDocument notebookDocument;

  /// The text documents that represent the content of a notebook cell.
  std::vector<TextDocumentItem> cellTextDocuments;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              DidOpenNotebookDocumentParams &result, llvm::json::Path path);

//===----------------------------------------------------------------------===//
// NotebookDocumentChangeEvent
//===----------------------------------------------------------------------===//

/// A change describing how to move a `NotebookCell` array from state S to S'.
struct NotebookCellArrayChange {
  /// The start offset of the cell that changed.
  uint64_t start;

  /// The deleted cells.
  uint64_t deleteCount;

  /// The new cells, if any.
  std::vector<NotebookCell> cells;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, NotebookCellArrayChange &result,
              llvm::json::Path path);

/// A change event for a notebook document.
struct NotebookDocumentChangeEvent {
  /// Changes to the cell structure to add or remove cells.
  struct CellsStructure {
    /// The change to the cell array.
    NotebookCellArrayChange array;

    /// Additional opened cell text documents.
    std::vector<TextDocumentItem> didOpen;

    /// Additional closed cell text documents.
    std::vector<TextDocumentIdentifier> didClose;
  };

  /// Changes to the text content of notebook cells.
  struct CellsTextContent {
    VersionedTextDocumentIdentifier document;
    std::vector<TextDocumentContentChangeEvent> changes;
  };

  /// Changes to cells.
  struct Cells {
    /// Changes to the cell structure to add or remove cells.
    std::optional<CellsStructure> structure;

    /// Changes to notebook cells properties like its kind, execution summary or
    /// metadata.
    std::vector<NotebookCell> data;

    /// Changes to the text content of notebook cells.
    std::vector<CellsTextContent> textContent;
  };

  /// Changes to cells.
  std::optional<Cells> cells;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              NotebookDocumentChangeEvent::CellsStructure &result,
              llvm::json::Path path);
bool fromJSON(const llvm::json::Value &value,
              NotebookDocumentChangeEvent::CellsTextContent &result,
              llvm::json::Path path);
bool fromJSON(const llvm::json::Value &value,
              NotebookDocumentChangeEvent::Cells &result,
              llvm::json::Path path);
bool fromJSON(const llvm::json::Value &value,
              NotebookDocumentChangeEvent &result, llvm::json::Path path);

//===----------------------------------------------------------------------===//
// DidChangeNotebookDocumentParams
//===----------------------------------------------------------------------===//

struct DidChangeNotebookDocumentParams {
  /// The notebook document that got opened.
  VersionedNotebookDocumentIdentifier notebookDocument;

  /// The actual changes to the notebook document.
  ///
  /// The change describes single state change to the notebook document.
  /// So it moves a notebook document, its cells and its cell text document
  /// contents from state S to S'.
  ///
  /// To mirror the content of a notebook using change events use the
  /// following approach:
  /// - start with the same initial content
  /// - apply the 'notebookDocument/didChange' notifications in the order
  ///   you receive them.
  NotebookDocumentChangeEvent change;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              DidChangeNotebookDocumentParams &result, llvm::json::Path path);

//===----------------------------------------------------------------------===//
// DidCloseNotebookDocumentParams
//===----------------------------------------------------------------------===//

/// The params sent in a close notebook document notification.
struct DidCloseNotebookDocumentParams {
  /// The notebook document that got closed.
  NotebookDocumentIdentifier notebookDocument;

  /// The text documents that represent the content of a notebook cell that got
  /// closed.
  std::vector<TextDocumentIdentifier> cellTextDocuments;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              DidCloseNotebookDocumentParams &result, llvm::json::Path path);

} // namespace lsp
} // namespace mlir

#endif
