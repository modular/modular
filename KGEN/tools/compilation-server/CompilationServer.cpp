//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "CompilationServer.h"
#include "LLVMServer.h"
#include "Protocol.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/Error.h"

#define DEBUG_TYPE "compilation-server"

namespace json = llvm::json;
using namespace mlir;
using namespace mlir::lsp;
using namespace M::KGEN::CSP;

//===----------------------------------------------------------------------===//
// CompilationServer
//===----------------------------------------------------------------------===//

namespace {
struct CompilationServer {
  CompilationServer(std::unique_ptr<LLVMServer> server)
      : llvmServer(std::move(server)) {}

  //===--------------------------------------------------------------------===//
  // Initialization

  void onInitialize(const NoParams &params, Callback<llvm::json::Value> reply);
  void onInitialized(const InitializedParams &params);
  void onShutdown(const NoParams &params, Callback<std::nullptr_t> reply);

  //===--------------------------------------------------------------------===//
  // Compilation

  void onEmitArchive(const EmitArchiveParams &params,
                     Callback<llvm::json::Value> reply);
  void onEchoMLIR(const MLIRModule &params, Callback<llvm::json::Value>);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  std::unique_ptr<LLVMServer> llvmServer;

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

//===--------------------------------------------------------------------===//
// Compilation

void CompilationServer::onEmitArchive(const EmitArchiveParams &params,
                                      Callback<llvm::json::Value> reply) {
  std::string result = llvmServer->emitArchive(params);

  ObjectArchive value;
  value.archive = std::move(result);
  reply(value);
}

void CompilationServer::onEchoMLIR(const MLIRModule &params,
                                   Callback<llvm::json::Value> reply) {
  std::string result = llvmServer->echoMLIR(params.module);
  MLIRModule value;
  value.module = std::move(result);
  reply(value);
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

mlir::LogicalResult M::KGEN::runCompilationServer(JSONTransport &transport,
                                                  bool singleThreaded) {
  MessageHandler messageHandler(transport);

  ErrorOr<std::unique_ptr<LLVMServer>> serverOr =
      LLVMServer::create(singleThreaded);
  if (serverOr.isError()) {
    auto error = llvm::make_error<llvm::StringError>(
        serverOr.getError(), llvm::inconvertibleErrorCode());
    Logger::error("Error creating LLVM server: {0}", error);
    llvm::consumeError(std::move(error));
    return failure();
  }
  CompilationServer compilationServer(serverOr.takeValue());

  // Initialization
  messageHandler.method("initialize", &compilationServer,
                        &CompilationServer::onInitialize);
  messageHandler.notification("initialized", &compilationServer,
                              &CompilationServer::onInitialized);
  messageHandler.method("shutdown", &compilationServer,
                        &CompilationServer::onShutdown);
  messageHandler.method("emitArchive", &compilationServer,
                        &CompilationServer::onEmitArchive);
  messageHandler.method("echoMLIR", &compilationServer,
                        &CompilationServer::onEchoMLIR);

  // Run the main loop of the transport.
  if (llvm::Error error = transport.run(messageHandler)) {
    Logger::error("Transport error: {0}", error);
    llvm::consumeError(std::move(error));
    return failure();
  }
  return success(compilationServer.shutdownRequestReceived);
}
