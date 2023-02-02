//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_LITLSPSERVER_LITSERVER_H
#define KGEN_TOOLS_LITLSPSERVER_LITSERVER_H

#include "Support/LLVMForwardDecls.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"

namespace M::KGEN::LIT {

/// This class implements all of the LIT related functionality necessary for a
/// language server. This class allows for keeping the LIT specific logic
/// separate from the logic that involves LSP server/client communication.
class LITServer {
public:
  LITServer();
  ~LITServer();

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

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace M::KGEN::LIT

#endif // KGEN_TOOLS_LITLSPSERVER_LITSERVER_H
