//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Config.h"
#include "mlir/Pass/PassManager.h"

void M::configurePassManager(mlir::PassManager &mgr) {
#ifdef MODULAR_PRODUCTION
  mgr.enableVerifier(false);
#endif
}
