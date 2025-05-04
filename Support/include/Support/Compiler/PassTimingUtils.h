//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_PASSTIMINGUTILS_H
#define SUPPORT_COMPILER_PASSTIMINGUTILS_H

#include "Support/ErrorOr.h"
#include "mlir/Pass/PassManager.h"

namespace M {
/// Enables pass timing on the `PassManger` object and dumps the tree structured
/// JSON to a temp file in `outDir`. The ostream object is initialized in the
/// function
ErrorOrSuccess
configureMLIRPassTimingJSONOutput(mlir::PassManager &pm, llvm::StringRef outDir,
                                  llvm::StringRef passPipelineName);
} // namespace M

#endif // SUPPORT_COMPILER_PASSTIMINGUTILS_H
