//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/PassManagerConfigOptions.h"
#include "Support/Config.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

ErrorOrSuccess
PassManagerConfigOptions::configurePassManager(mlir::PassManager &pm) const {
  M::configurePassManager(pm);

  if (applyPassManagerCLOptions) {
    if (failed(mlir::applyPassManagerCLOptions(pm)))
      return Error("applyPassManagerCLOptions failed during configuring");
  }

  if (enableTiming) {
    if (timingScope)
      pm.enableTiming(*timingScope);
    else
      pm.enableTiming();
  }

  if (crashReproducerOptions.enable) {
    pm.enableCrashReproducerGeneration(
        crashReproducerOptions.inputFileName + ".repro.mlir",
        crashReproducerOptions.enableLocalMLIRReproducer);
  }

  if (irPrintingOptions.enable) {
    pm.enableIRPrinting(
        [&](mlir::Pass *pass, mlir::Operation *) -> bool {
          return pass->getName() == irPrintingOptions.passName;
        },
        [&](mlir::Pass *, mlir::Operation *) -> bool {
          return irPrintingOptions.shouldPrintAfterPass;
        },
        irPrintingOptions.printModuleScope,
        irPrintingOptions.printAfterOnlyOnChange,
        irPrintingOptions.printAfterOnlyOnFailure, *irPrintingOptions.out,
        irPrintingOptions.opPrintingFlags);
  }

  return {};
}
