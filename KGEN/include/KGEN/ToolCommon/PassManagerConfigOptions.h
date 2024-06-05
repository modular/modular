//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_PASSMANAGERCONFIGOPTIONS_H
#define KGEN_TOOLCOMMON_PASSMANAGERCONFIGOPTIONS_H

#include "Support/LogicalResult.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include <string>

namespace M::KGEN {
struct PassManagerConfigOptions {
  struct CrashReproducerOptions {
    bool enable = false;
    bool enableLocalMLIRReproducer = false;
    std::string inputFileName;
  };

  struct IRPrintingOptions {
    bool enable = false;
    std::string passName;
    bool shouldPrintAfterPass = false;
    bool printModuleScope = false;
    bool printAfterOnlyOnChange = false;
    bool printAfterOnlyOnFailure = false;
    llvm::raw_ostream *out;
    mlir::OpPrintingFlags opPrintingFlags = mlir::OpPrintingFlags();
  };

  CrashReproducerOptions crashReproducerOptions;
  bool enableTiming = false;
  mlir::TimingScope *timingScope = nullptr;
  IRPrintingOptions irPrintingOptions;
  bool applyPassManagerCLOptions = false;
  std::optional<std::string> operationName;

  LogicalResult configurePassManager(mlir::PassManager &pm) const;
};

} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_PASSMANAGERCONFIGOPTIONS_H
