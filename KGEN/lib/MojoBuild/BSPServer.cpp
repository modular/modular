//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "BSPServer.h"

#include "KGEN/MojoBuild/Protocol.h"

#include "mlir/Tools/lsp-server-support/Logging.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace M::Build;
using namespace mlir;

BSPServer::BSPServer()
    : transport(stdin, llvm::outs(), lsp::JSONStreamStyle::Delimited,
                /*prettyOutput=*/true),
      messageHandler(transport) {
  messageHandler.method("build/initialize", this,
                        &BSPServer::onBuildInitialize);
}

mlir::LogicalResult BSPServer::run() {
  if (llvm::Error error = transport.run(messageHandler)) {
    lsp::Logger::error("server transport error: {0}", error);
    llvm::consumeError(std::move(error));
    return failure();
  }
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
  callback(InitializeBuildResult{"mojo-build-server"});
}
