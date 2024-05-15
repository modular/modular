//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "BSPServer.h"

#include "Config/Version.h"
#include "KGEN/MojoBuild/Protocol.h"
#include "Support/ErrorOr.h"

#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>

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

  while (!serverResult.has_value()) {
    ErrorOrSuccess result = runTransport(transport, messageHandler);
    if (failed(result))
      return result.takeError();
  }

  // If the server needs to be shut down due to an error, exit immediately.
  if (failed(*serverResult))
    return Error("server shut down due to an error");

  // Otherwise, after receiving a shutdown request, run the transport one last
  // time, to process any communication related to shutting down.
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
  std::error_code ec;
  bool exists = std::filesystem::exists(params.rootUri, ec);
  if (ec)
    return callback(error(
        llvm::formatv("server could not be initialized, an error occurred when "
                      "accessing rootUri '{0}': {1}",
                      params.rootUri, ec.message()),
        lsp::ErrorCode::InvalidParams));
  if (!exists)
    return callback(error(
        llvm::formatv(
            "server could not be initialized, rootUri '{0}' does not exist",
            params.rootUri),
        lsp::ErrorCode::InvalidParams));

  callback(InitializeBuildResult{"mojo-build-server", getModularVersionString(),
                                 /*bspVersion=*/"2.2.0",
                                 BuildServerCapabilities{}});
}

void BSPServer::onBuildShutdown(const NoParams &params,
                                mlir::lsp::Callback<NoParams> callback) {
  serverResult = success();
  callback(NoParams{});
}

llvm::Error BSPServer::error(Twine message, mlir::lsp::ErrorCode code) {
  serverResult = failure();
  return llvm::make_error<lsp::LSPError>(message.str(), code);
}
