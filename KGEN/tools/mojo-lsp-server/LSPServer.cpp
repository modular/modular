//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LSPServer.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "MojoServer.h"
#include "Protocol.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringMap.h"
#include <optional>

#define DEBUG_TYPE "mojo-lsp-server"

using namespace mlir::lsp;
using namespace M;
using namespace M::Mojo::LSP;

//===----------------------------------------------------------------------===//
// LSPServer
//===----------------------------------------------------------------------===//

namespace {
struct LSPServer {
  LSPServer(MojoServer &server, JSONTransport &transport)
      : server(server), transport(transport) {}

  //===--------------------------------------------------------------------===//
  // Initialization

  void onInitialize(const InitializeParams &params,
                    Callback<llvm::json::Value> reply);
  void onInitialized(const InitializedParams &params);
  void onShutdown(const NoParams &params, Callback<std::nullptr_t> reply);

  //===--------------------------------------------------------------------===//
  // Document Change

  void onDocumentDidOpen(const DidOpenTextDocumentParams &params);
  void onDocumentDidClose(const DidCloseTextDocumentParams &params);
  void onDocumentDidChange(const DidChangeTextDocumentParams &params);

  void onNotebookDocumentDidOpen(const DidOpenNotebookDocumentParams &params);
  void onNotebookDocumentDidClose(const DidCloseNotebookDocumentParams &params);
  void
  onNotebookDocumentDidChange(const DidChangeNotebookDocumentParams &params);

  //===--------------------------------------------------------------------===//
  // Code Action

  void onCodeAction(const CodeActionParams &params,
                    Callback<llvm::json::Value> reply);

  //===--------------------------------------------------------------------===//
  // Language Features

  void onCompletion(const CompletionParams &params,
                    Callback<CompletionList> reply);

  void onDefinition(const TextDocumentPositionParams &params,
                    Callback<llvm::json::Value> reply);

  void onHover(const TextDocumentPositionParams &params,
               Callback<llvm::json::Value> reply);

  void onSignatureHelp(const TextDocumentPositionParams &params,
                       Callback<SignatureHelp2> reply);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  MojoServer &server;
  JSONTransport &transport;

  /// Used to indicate that the 'shutdown' request was received from the
  /// Language Server client.
  bool shutdownRequestReceived = false;
};
} // namespace

//===----------------------------------------------------------------------===//
// Initialization

void LSPServer::onInitialize(const InitializeParams &params,
                             Callback<llvm::json::Value> reply) {
  using JSONArray = llvm::json::Array;
  using JSONObject = llvm::json::Object;

  // Send a response with the capabilities of this server.
  JSONObject serverCaps{
      {"completionProvider",
       JSONObject{
           {"allCommitCharacters", {"\t", "."}},
           {"resolveProvider", false},
           {"triggerCharacters", {"."}},
       }},
      {"signatureHelpProvider",
       llvm::json::Object{
           {"triggerCharacters", {"(", "[", ","}},
       }},
      {"definitionProvider", true},
      {"hoverProvider", true},
      {"notebookDocumentSync",
       JSONObject{{
           "notebookSelector",
           JSONArray{JSONObject{
               {"notebook", JSONObject{{"scheme", "file"},
                                       {"notebookType", "jupyter-notebook"}}},
               {"cells", JSONArray{JSONObject{{"language", "mojo"}}}},
           }},
       }}},
      {
          "textDocumentSync",
          JSONObject{
              {"openClose", true},
              {"change", (int)TextDocumentSyncKind::Incremental},
              {"save", true},
          },
      },
  };

  // Per LSP, codeActionProvider can be either boolean or CodeActionOptions.
  // CodeActionOptions is only valid if the client supports action literal
  // via textDocument.codeAction.codeActionLiteralSupport.
  serverCaps["codeActionProvider"] =
      params.capabilities.codeActionStructure
          ? llvm::json::Object{{"codeActionKinds",
                                {CodeAction::kQuickFix, CodeAction::kRefactor,
                                 CodeAction::kInfo}}}
          : llvm::json::Value(true);

  llvm::json::Object result{
      {{"serverInfo",
        llvm::json::Object{{"name", "mojo-lsp-server"}, {"version", "0.0.1"}}},
       {"capabilities", std::move(serverCaps)}}};
  reply(std::move(result));
}
void LSPServer::onInitialized(const InitializedParams &) {}
void LSPServer::onShutdown(const NoParams &, Callback<std::nullptr_t> reply) {
  server.shutdown();
  shutdownRequestReceived = true;
  reply(nullptr);
}

//===----------------------------------------------------------------------===//
// Document Change

void LSPServer::onDocumentDidOpen(const DidOpenTextDocumentParams &params) {
  server.addDocument(params.textDocument.uri,
                     std::string(params.textDocument.text),
                     params.textDocument.version);
}
void LSPServer::onDocumentDidClose(const DidCloseTextDocumentParams &params) {
  server.removeDocument(params.textDocument.uri);
}
void LSPServer::onDocumentDidChange(const DidChangeTextDocumentParams &params) {
  server.updateDocument(params.textDocument.uri, params.contentChanges,
                        params.textDocument.version);
}

void LSPServer::onNotebookDocumentDidOpen(
    const DidOpenNotebookDocumentParams &params) {
  server.addNotebookDocument(
      params.notebookDocument.uri, params.notebookDocument.cells,
      params.notebookDocument.version, params.cellTextDocuments);
}
void LSPServer::onNotebookDocumentDidClose(
    const DidCloseNotebookDocumentParams &params) {
  server.removeNotebookDocument(params.notebookDocument.uri,
                                params.cellTextDocuments);
}
void LSPServer::onNotebookDocumentDidChange(
    const DidChangeNotebookDocumentParams &params) {
  server.updateNotebookDocument(params.notebookDocument.uri,
                                params.notebookDocument.version, params.change);
}

//===----------------------------------------------------------------------===//
// Code Action

void LSPServer::onCodeAction(const CodeActionParams &params,
                             Callback<llvm::json::Value> reply) {
  URIForFile uri = params.textDocument.uri;

  // Check whether a particular CodeActionKind is included in the response.
  auto isKindAllowed = [only(params.context.only)](StringRef kind) {
    if (only.empty())
      return true;
    return llvm::any_of(only, [&](StringRef base) {
      return kind.consume_front(base) && (kind.empty() || kind.startswith("."));
    });
  };

  // We provide a code action for fixes on the specified diagnostics.
  if (!isKindAllowed(CodeAction::kQuickFix))
    return reply(std::vector<CodeAction>());

  server.getCodeActions(
      uri, params.range.start, params.context,
      [reply = std::move(reply)](std::vector<CodeAction> actions) mutable {
        reply(std::move(actions));
      });
}

//===----------------------------------------------------------------------===//
// Language Features

void LSPServer::onCompletion(const CompletionParams &params,
                             Callback<CompletionList> reply) {
  server.onCodeCompletion(
      params.textDocument.uri, params.position,
      [reply = std::move(reply)](CompletionList list) mutable {
        reply(std::move(list));
      });
}

void LSPServer::onDefinition(const TextDocumentPositionParams &params,
                             Callback<llvm::json::Value> reply) {
  server.onDefinition(params.textDocument.uri, params.position,
                      [reply = std::move(reply)](
                          std::vector<mlir::lsp::Location> locations) mutable {
                        reply(std::move(locations));
                      });
}

void LSPServer::onHover(const TextDocumentPositionParams &params,
                        Callback<llvm::json::Value> reply) {
  server.onHover(
      params.textDocument.uri, params.position,
      [reply = std::move(reply)](std::optional<Hover> hover) mutable {
        reply(std::move(hover));
      });
}

void LSPServer::onSignatureHelp(const TextDocumentPositionParams &params,
                                Callback<SignatureHelp2> reply) {
  server.getSignatureHelp(
      params.textDocument.uri, params.position,
      [repl = std::move(reply)](SignatureHelp help) mutable {
        // TODO: Remove this when the changes to SignatureHelp are upstreamed.
        SignatureHelp2 help2;
        help2.activeParameter = help.activeParameter;
        help2.activeSignature = help.activeSignature;
        for (auto &sig : help.signatures) {
          SignatureInformation2 sig2;
          sig2.label = sig.label;
          sig2.documentation =
              MarkupContent{MarkupKind::Markdown, sig.documentation};
          sig2.parameters = std::move(sig.parameters);
          help2.signatures.push_back(sig2);
        }

        repl(std::move(help2));
      });
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

mlir::LogicalResult
M::KGEN::LIT::runMojoLSPServer(JSONTransport &transport,
                               std::unique_ptr<LLCL::WorkQueue> workQueue,
                               bool waitOnShutdown) {
  MessageHandler messageHandler(transport);
  MojoServer server(
      std::move(workQueue), waitOnShutdown,
      messageHandler.outgoingNotification<PublishDiagnosticsParams>(
          "textDocument/publishDiagnostics"));
  LSPServer lspServer(server, transport);

  // Initialization
  messageHandler.method("initialize", &lspServer, &LSPServer::onInitialize);
  messageHandler.notification("initialized", &lspServer,
                              &LSPServer::onInitialized);
  messageHandler.method("shutdown", &lspServer, &LSPServer::onShutdown);

  // Document Changes
  messageHandler.notification("notebookDocument/didOpen", &lspServer,
                              &LSPServer::onNotebookDocumentDidOpen);
  messageHandler.notification("notebookDocument/didClose", &lspServer,
                              &LSPServer::onNotebookDocumentDidClose);
  messageHandler.notification("notebookDocument/didChange", &lspServer,
                              &LSPServer::onNotebookDocumentDidChange);
  messageHandler.notification("textDocument/didOpen", &lspServer,
                              &LSPServer::onDocumentDidOpen);
  messageHandler.notification("textDocument/didClose", &lspServer,
                              &LSPServer::onDocumentDidClose);
  messageHandler.notification("textDocument/didChange", &lspServer,
                              &LSPServer::onDocumentDidChange);

  // Code Action
  messageHandler.method("textDocument/codeAction", &lspServer,
                        &LSPServer::onCodeAction);

  // Language Features
  messageHandler.method("textDocument/completion", &lspServer,
                        &LSPServer::onCompletion);
  messageHandler.method("textDocument/definition", &lspServer,
                        &LSPServer::onDefinition);
  messageHandler.method("textDocument/hover", &lspServer, &LSPServer::onHover);
  messageHandler.method("textDocument/signatureHelp", &lspServer,
                        &LSPServer::onSignatureHelp);

  // Run the main loop of the transport.
  if (llvm::Error error = transport.run(messageHandler)) {
    Logger::error("Transport error: {0}", error);
    llvm::consumeError(std::move(error));
    return failure();
  }
  return success(lspServer.shutdownRequestReceived);
}
