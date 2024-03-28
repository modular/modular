//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoBuild/Server.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"

MODULAR_EXPORT int mojoBuildServerMain() {
  mlir::lsp::JSONTransport transport(stdin, llvm::outs(),
                                     mlir::lsp::JSONStreamStyle::Delimited,
                                     /*prettyOutput=*/true);
  mlir::lsp::MessageHandler messageHandler(transport);

  if (llvm::Error error = transport.run(messageHandler)) {
    mlir::lsp::Logger::error("Transport error: {0}", error);
    llvm::consumeError(std::move(error));
    return 1;
  }

  return 0;
}
