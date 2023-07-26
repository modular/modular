//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LSPServer.h"
#include "MojoServer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringMap.h"
#include <optional>

#define DEBUG_TYPE "mojo-lsp-server"

using namespace mlir::lsp;
using namespace M;
using namespace M::KGEN::LIT;

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
  // Send a response with the capabilities of this server.
  llvm::json::Object serverCaps{
      {"completionProvider",
       llvm::json::Object{
           {"allCommitCharacters",
            {"\t", "(", ")", "[", "]", "{",  "}", "<", ">",
             ":",  ";", ",", "+", "-", "/",  "*", "%", "^",
             "&",  "#", "?", ".", "=", "\"", "'", "|"}},
           {"resolveProvider", false},
           {"triggerCharacters", {"."}},
       }},
      {"definitionProvider", true},
      {"hoverProvider", true},
      {"textDocumentSync",
       llvm::json::Object{
           {"openClose", true},
           {"change", (int)TextDocumentSyncKind::Incremental},
           {"save", true},
       }},
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
  shutdownRequestReceived = true;
  reply(nullptr);
}

//===----------------------------------------------------------------------===//
// Document Change

void LSPServer::onDocumentDidOpen(const DidOpenTextDocumentParams &params) {
  server.addDocument(params.textDocument.uri, params.textDocument.text,
                     params.textDocument.version);
}
void LSPServer::onDocumentDidClose(const DidCloseTextDocumentParams &params) {
  server.removeDocument(params.textDocument.uri);
}
void LSPServer::onDocumentDidChange(const DidChangeTextDocumentParams &params) {
  server.updateDocument(params.textDocument.uri, params.contentChanges,
                        params.textDocument.version);
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
  std::vector<CodeAction> actions;
  if (isKindAllowed(CodeAction::kQuickFix))
    server.getCodeActions(uri, params.range.start, params.context, actions);
  reply(std::move(actions));
}

//===----------------------------------------------------------------------===//
// Language Features

void LSPServer::onCompletion(const CompletionParams &params,
                             Callback<CompletionList> reply) {
  reply(server.getCodeCompletion(params.textDocument.uri, params.position));
}

void LSPServer::onDefinition(const TextDocumentPositionParams &params,
                             Callback<llvm::json::Value> reply) {
  reply(server.onDefinition(params.textDocument.uri, params.position));
}

void LSPServer::onHover(const TextDocumentPositionParams &params,
                        Callback<llvm::json::Value> reply) {
  reply(server.onHover(params.textDocument.uri, params.position));
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

mlir::LogicalResult M::KGEN::LIT::runMojoLSPServer(JSONTransport &transport) {
  MessageHandler messageHandler(transport);
  MojoServer server(
      messageHandler.outgoingNotification<PublishDiagnosticsParams>(
          "textDocument/publishDiagnostics"));
  LSPServer lspServer(server, transport);

  // Initialization
  messageHandler.method("initialize", &lspServer, &LSPServer::onInitialize);
  messageHandler.notification("initialized", &lspServer,
                              &LSPServer::onInitialized);
  messageHandler.method("shutdown", &lspServer, &LSPServer::onShutdown);

  // Document Changes
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

  // Run the main loop of the transport.
  if (llvm::Error error = transport.run(messageHandler)) {
    Logger::error("Transport error: {0}", error);
    llvm::consumeError(std::move(error));
    return failure();
  }
  return success(lspServer.shutdownRequestReceived);
}
