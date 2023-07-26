//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
#define KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H

#include "Support/LLVMForwardDecls.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "llvm/ADT/FunctionExtras.h"

namespace M::KGEN::LIT {
using SendDiagnosticsFn =
    llvm::unique_function<void(const mlir::lsp::PublishDiagnosticsParams &)>;

/// This class implements all of the Mojo related functionality necessary for a
/// language server. This class allows for keeping the Mojo specific logic
/// separate from the logic that involves LSP server/client communication.
class MojoServer {
public:
  MojoServer(SendDiagnosticsFn sendDiagnosticsFn);
  ~MojoServer();

  /// Add the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document will be added to `diagnostics`.
  void addDocument(const mlir::lsp::URIForFile &uri, StringRef contents,
                   int64_t version);

  /// Update the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document will be added to `diagnostics`.
  void
  updateDocument(const mlir::lsp::URIForFile &uri,
                 ArrayRef<mlir::lsp::TextDocumentContentChangeEvent> changes,
                 int64_t version);

  /// Remove the document with the given uri.
  void removeDocument(const mlir::lsp::URIForFile &uri);

  /// Get the set of code actions within the file.
  void getCodeActions(const mlir::lsp::URIForFile &uri,
                      const mlir::lsp::Range &pos,
                      const mlir::lsp::CodeActionContext &context,
                      std::vector<mlir::lsp::CodeAction> &actions);

  /// Get the code completion list for the position within the given file.
  mlir::lsp::CompletionList
  getCodeCompletion(const mlir::lsp::URIForFile &uri,
                    const mlir::lsp::Position &completePos);

  /// Get the location of identifier of the declaration of the symbol that
  /// contains the given position.
  std::optional<mlir::lsp::Location>
  onDefinition(const mlir::lsp::URIForFile &uri,
               const mlir::lsp::Position &pos);

  /// Get a `Hover` element corresponding to the given document position.
  std::optional<mlir::lsp::Hover> onHover(const mlir::lsp::URIForFile &uri,
                                          const mlir::lsp::Position &pos);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace M::KGEN::LIT

#endif // KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
