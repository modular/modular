//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "BSPServer.h"
#include "Protocol.h"

using namespace M;
using namespace M::Build;

void BSPServer::onBuildInitialize(
    const InitializeBuildParams &params,
    mlir::lsp::Callback<llvm::json::Value> callback) {
  llvm::json::Object result{{"displayName", "mojo-build-server"}};
  callback(std::move(result));
}
