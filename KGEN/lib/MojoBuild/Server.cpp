//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoBuild/Server.h"
#include "KGEN/MojoBuild/Protocol.h"

#include "BSPServer.h"

#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/lsp-server-support/Logging.h"

using namespace M;
using namespace M::Build;

MODULAR_EXPORT int mojoBuildServerMain() {
  mlir::lsp::Logger::setLogLevel(mlir::lsp::Logger::Level::Debug);

  BSPServer bspServer;
  return succeeded(bspServer.run()) ? 0 : 1;
}
