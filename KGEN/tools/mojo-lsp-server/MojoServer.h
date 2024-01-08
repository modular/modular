//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
#define KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H

#include "Protocol.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/FunctionExtras.h"

namespace M::LLCL {
class WorkQueue;
} // namespace M::LLCL

namespace M::Mojo::LSP {
using SendDiagnosticsFn =
    llvm::unique_function<void(const mlir::lsp::PublishDiagnosticsParams &)>;
using SendDiagnosticsFnRef =
    function_ref<void(const mlir::lsp::PublishDiagnosticsParams &)>;
template <typename T>
using OnResultFn = llvm::unique_function<void(T)>;

/// This class implements all of the Mojo related functionality necessary for a
/// language server. This class allows for keeping the Mojo specific logic
/// separate from the logic that involves LSP server/client communication.
class MojoServer {
public:
  MojoServer(bool singleThreaded, bool waitOnShutdown,
             SendDiagnosticsFn sendDiagnosticsFn, bool parseStdlib);
  ~MojoServer();

  /// Begin the shutdown sequence for the server.
  void shutdown();

  //===--------------------------------------------------------------------===//
  // Document Management
  //===--------------------------------------------------------------------===//

  /// Add the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document will be added to `diagnostics`.
  void addDocument(const mlir::lsp::URIForFile &uri, std::string &&contents,
                   int64_t version);

  /// Update the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document will be added to `diagnostics`.
  void
  updateDocument(const mlir::lsp::URIForFile &uri,
                 ArrayRef<mlir::lsp::TextDocumentContentChangeEvent> changes,
                 int64_t version);

  /// Remove the document with the given uri.
  void removeDocument(const mlir::lsp::URIForFile &uri);

  //===--------------------------------------------------------------------===//
  // Notebook Document Management
  //===--------------------------------------------------------------------===//

  /// Add the notebook document, with the provided `version`, at the given URI.
  /// Any diagnostics emitted for this document will be added to `diagnostics`.
  void addNotebookDocument(const mlir::lsp::URIForFile &uri,
                           ArrayRef<mlir::lsp::NotebookCell> cells,
                           int64_t version,
                           ArrayRef<mlir::lsp::TextDocumentItem> cellDocuments);

  /// Remove the notebook document with the given uri.
  void removeNotebookDocument(
      const mlir::lsp::URIForFile &uri,
      ArrayRef<mlir::lsp::TextDocumentIdentifier> cellDocuments);

  /// Update the document, with the provided `version`, at the given URI.
  void
  updateNotebookDocument(const mlir::lsp::URIForFile &uri, int64_t version,
                         const mlir::lsp::NotebookDocumentChangeEvent &change);

  //===--------------------------------------------------------------------===//
  // Queries
  //===--------------------------------------------------------------------===//

  /// Get the set of code actions within the file.
  void
  getCodeActions(const mlir::lsp::URIForFile &uri, const mlir::lsp::Range &pos,
                 const mlir::lsp::CodeActionContext &context,
                 OnResultFn<std::vector<mlir::lsp::CodeAction>> onActionsFn);

  /// Get the code completion list for the position within the given file.
  void onCodeCompletion(const mlir::lsp::URIForFile &uri,
                        const mlir::lsp::Position &completePos,
                        OnResultFn<mlir::lsp::CompletionList> onCompletionFn);

  /// Get the identifier location of the symbol declarations that contain the
  /// given position.
  void
  onDefinition(const mlir::lsp::URIForFile &uri, const mlir::lsp::Position &pos,
               OnResultFn<std::vector<mlir::lsp::Location>> onDefinitionFn);

  /// Find all of the document symbols within the given file.
  void onDocumentSymbol(
      const mlir::lsp::URIForFile &uri,
      OnResultFn<std::vector<mlir::lsp::DocumentSymbol>> onSymbolsFn);

  /// Find all of the folding ranges within the given file.
  void onFoldingRange(
      const mlir::lsp::URIForFile &uri,
      OnResultFn<std::vector<mlir::lsp::FoldingRange>> onFoldingRangeFn);

  /// Get a `Hover` element corresponding to the given document position.
  void onHover(const mlir::lsp::URIForFile &uri, const mlir::lsp::Position &pos,
               OnResultFn<std::optional<mlir::lsp::Hover>> onHoverFn);

  /// Get inlay hints for the given document range.
  void onInlayHint(const mlir::lsp::URIForFile &uri,
                   const mlir::lsp::Range &range,
                   OnResultFn<std::vector<mlir::lsp::InlayHint>> onInlayHint);

  // Get the references of the symbol in the given location.
  void onReferences(const mlir::lsp::URIForFile &uri,
                    const mlir::lsp::Position &position,
                    bool includeDeclaration,
                    OnResultFn<std::vector<mlir::lsp::Location>> onReferences);

  /// Get the semantic tokens for the given document.
  void onSemanticTokens(
      const mlir::lsp::URIForFile &uri,
      OnResultFn<std::optional<mlir::lsp::SemanticTokens>> onSemanticTokens);

  /// Get the delta of semantic tokens for the given document compared to the
  /// tokens at the given identifier (representing a previous result).
  void onSemanticTokensDelta(
      const mlir::lsp::URIForFile &uri, StringRef prevId,
      OnResultFn<std::optional<mlir::lsp::SemanticTokensOrDelta>>
          onSemanticTokens);

  /// Get the signature help for the position within the given document.
  void getSignatureHelp(const mlir::lsp::URIForFile &uri,
                        const mlir::lsp::Position &pos,
                        OnResultFn<mlir::lsp::SignatureHelp> onHelpFn);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace M::Mojo::LSP

#endif // KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
