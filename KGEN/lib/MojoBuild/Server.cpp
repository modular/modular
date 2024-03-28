//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoBuild/Server.h"

#include "BSPServer.h"
#include "Protocol.h"

#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/Program.h"

using namespace M;
using namespace M::Build;

MODULAR_EXPORT int mojoBuildServerMain() {
  llvm::sys::ChangeStdinToBinary();
  mlir::lsp::JSONTransport transport(stdin, llvm::outs(),
                                     mlir::lsp::JSONStreamStyle::Delimited,
                                     /*prettyOutput=*/true);
  BSPServer bspServer;

  mlir::lsp::MessageHandler messageHandler(transport);
  messageHandler.method("build/initialize", &bspServer,
                        &BSPServer::onBuildInitialize);

  if (llvm::Error error = transport.run(messageHandler)) {
    mlir::lsp::Logger::error("Transport error: {0}", error);
    llvm::consumeError(std::move(error));
    return 1;
  }

  return 0;
}
