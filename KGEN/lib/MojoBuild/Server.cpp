//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoBuild/Server.h"
#include "BSPServer.h"
#include "KGEN/MojoBuild/Protocol.h"

#include "mlir/Tools/lsp-server-support/Logging.h"

using namespace M;
using namespace M::Build;
using namespace mlir;

MODULAR_EXPORT int mojoBuildServerMain(bool debug) {
  lsp::Logger::setLogLevel(mlir::lsp::Logger::Level::Debug);

  BSPServer bspServer(debug);
  ErrorOrSuccess result = bspServer.run();
  if (result.isError()) {
    lsp::Logger::error("server did not shut down properly: {0}",
                       result.getError());
    llvm::errs() << "mojo-build-server: error: " << result.getError() << '\n';
    return 1;
  }

  return 0;
}
