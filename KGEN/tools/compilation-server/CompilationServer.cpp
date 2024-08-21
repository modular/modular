//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "CompilationServer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include <optional>

#define DEBUG_TYPE "compilation-server"

using namespace mlir::lsp;
using namespace M;

//===----------------------------------------------------------------------===//
// CompilationServer
//===----------------------------------------------------------------------===//

namespace {
struct CompilationServer {
  CompilationServer() {}

  //===--------------------------------------------------------------------===//
  // Initialization

  void onInitialize(const NoParams &params, Callback<llvm::json::Value> reply);
  void onInitialized(const InitializedParams &params);
  void onShutdown(const NoParams &params, Callback<std::nullptr_t> reply);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// Used to indicate that the 'shutdown' request was received from the
  /// Compilation Server client.
  bool shutdownRequestReceived = false;
};
} // namespace

//===----------------------------------------------------------------------===//
// Initialization

void CompilationServer::onInitialize(const NoParams &params,
                                     Callback<llvm::json::Value> reply) {
  using JSONValue = llvm::json::Value;

  // Send a 'hello' response to help with testing.
  JSONValue hello("hello");
  reply(hello);
}

void CompilationServer::onInitialized(const InitializedParams &) {}
void CompilationServer::onShutdown(const NoParams &,
                                   Callback<std::nullptr_t> reply) {
  shutdownRequestReceived = true;
  reply(nullptr);
}
//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

mlir::LogicalResult M::KGEN::runCompilationServer(JSONTransport &transport) {
  CompilationServer compilationServer;
  MessageHandler messageHandler(transport);

  // Initialization
  messageHandler.method("initialize", &compilationServer,
                        &CompilationServer::onInitialize);
  messageHandler.notification("initialized", &compilationServer,
                              &CompilationServer::onInitialized);
  messageHandler.method("shutdown", &compilationServer,
                        &CompilationServer::onShutdown);

  // Run the main loop of the transport.
  if (llvm::Error error = transport.run(messageHandler)) {
    Logger::error("Transport error: {0}", error);
    llvm::consumeError(std::move(error));
    return failure();
  }
  return success(compilationServer.shutdownRequestReceived);
}
