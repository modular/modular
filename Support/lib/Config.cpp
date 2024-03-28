//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Config.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

void M::configureMLIRContext(mlir::MLIRContext &ctx) {
#ifdef MODULAR_PRODUCTION
  ctx.printOpOnDiagnostic(false);
#endif
}

void M::configurePassManager(mlir::PassManager &mgr) {
#ifdef MODULAR_PRODUCTION
  mgr.enableVerifier(false);
#endif
}
