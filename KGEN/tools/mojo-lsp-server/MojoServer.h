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
  MojoServer(std::unique_ptr<LLCL::WorkQueue> workQueue, bool waitOnShutdown,
             SendDiagnosticsFn sendDiagnosticsFn);
  ~MojoServer();

  /// Begin the shutdown sequence for the server.
  void shutdown();

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

  /// Get a `Hover` element corresponding to the given document position.
  void onHover(const mlir::lsp::URIForFile &uri, const mlir::lsp::Position &pos,
               OnResultFn<std::optional<mlir::lsp::Hover>> onHoverFn);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace M::Mojo::LSP

#endif // KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
