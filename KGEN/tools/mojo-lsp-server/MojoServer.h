//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
#define KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H

#include "Support/LLVMForwardDecls.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"

namespace M::KGEN::LIT {

/// This class implements all of the Mojo related functionality necessary for a
/// language server. This class allows for keeping the Mojo specific logic
/// separate from the logic that involves LSP server/client communication.
class MojoServer {
public:
  MojoServer();
  ~MojoServer();

  /// Add the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document will be added to `diagnostics`.
  void addDocument(const mlir::lsp::URIForFile &uri, StringRef contents,
                   int64_t version,
                   std::vector<mlir::lsp::Diagnostic> &diagnostics);

  /// Update the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document will be added to `diagnostics`.
  void
  updateDocument(const mlir::lsp::URIForFile &uri,
                 ArrayRef<mlir::lsp::TextDocumentContentChangeEvent> changes,
                 int64_t version,
                 std::vector<mlir::lsp::Diagnostic> &diagnostics);

  /// Remove the document with the given uri. Returns the version of the removed
  /// document, or std::nullopt if the uri did not have a corresponding document
  /// within the server.
  std::optional<int64_t> removeDocument(const mlir::lsp::URIForFile &uri);

  /// Get the set of code actions within the file.
  void getCodeActions(const mlir::lsp::URIForFile &uri,
                      const mlir::lsp::Range &pos,
                      const mlir::lsp::CodeActionContext &context,
                      std::vector<mlir::lsp::CodeAction> &actions);

  /// Get a `Hover` element corresponding to the given document position.
  std::optional<mlir::lsp::Hover> onHover(const mlir::lsp::URIForFile &uri,
                                          const mlir::lsp::Position &pos);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace M::KGEN::LIT

#endif // KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
