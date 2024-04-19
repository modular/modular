//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "BSPServer.h"

#include "KGEN/MojoBuild/Protocol.h"
#include "Support/ErrorOr.h"

#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace M::Build;
using namespace mlir;

BSPServer::BSPServer(bool debug)
    : transport(stdin, llvm::outs(),
                debug ? lsp::JSONStreamStyle::Delimited
                      : lsp::JSONStreamStyle::Standard,
                /*prettyOutput=*/debug),
      messageHandler(transport) {
  messageHandler.method("build/initialize", this,
                        &BSPServer::onBuildInitialize);
  messageHandler.method("build/shutdown", this, &BSPServer::onBuildShutdown);
}

ErrorOrSuccess BSPServer::run() {
  auto runTransport =
      [](lsp::JSONTransport &transport,
         lsp::MessageHandler &messageHandler) -> ErrorOrSuccess {
    if (llvm::Error error = transport.run(messageHandler)) {
      llvm::consumeError(std::move(error));
      if (feof(stdin)) {
        clearerr(stdin);
      } else {
        lsp::Logger::error("server transport error: {0}", error);
        return Error(llvm::formatv("JSON transport error: {0}", error));
      }
    }
    return success();
  };

  while (!shutdownRequestReceived) {
    ErrorOrSuccess result = runTransport(transport, messageHandler);
    if (failed(result))
      return result.takeError();
  }
  // After receiving a shutdown request, run the transport one last time, to
  // process any communication related to shutting down.
  ErrorOrSuccess result = runTransport(transport, messageHandler);
  if (failed(result))
    return result.takeError();

  return success();
}

//===----------------------------------------------------------------------===//
// Request handlers
//===----------------------------------------------------------------------===//

void BSPServer::onBuildInitialize(
    const InitializeBuildParams &params,
    mlir::lsp::Callback<InitializeBuildResult> callback) {
  lsp::Logger::debug("<-- server build/initialize: displayName='{0}'",
                     params.displayName);
  // FIXME(#36902): Rather than respond via a notification, this should send a
  // reply via `callback` once MLIR's LSP supports doing so.
  transport.notify("build/initialize/reply",
                   InitializeBuildResult{"mojo-build-server"});
}

void BSPServer::onBuildShutdown(const NoParams &params,
                                mlir::lsp::Callback<std::nullptr_t> callback) {
  lsp::Logger::debug("<-- server build/shutdown");
  shutdownRequestReceived = true;
}
