//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Interface for the kgen-opt MLIR processing path.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_KGEN_OPT_MLIR_DRIVER_H
#define KGEN_TOOLS_KGEN_OPT_MLIR_DRIVER_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace M::KGEN::Tool {

/// Register all KGEN MLIR dialects, translations, and passes into \p registry.
/// Must be called before registerMLIRPathCLOptions().
void registerMLIRDialectsAndPasses(mlir::DialectRegistry &registry);

/// Create the AsyncRT context, register it with \p registry, and register all
/// MLIR-path CL options (pass pipeline, diagnostics, KGEN passes, …).
/// Scans argv for --asyncrt-single-thread before option parsing.
/// Returns false and prints an error if context creation fails.
/// Must be called before llvm::cl::ParseCommandLineOptions().
bool registerMLIRPathCLOptions(mlir::DialectRegistry &registry, int argc,
                               char **argv);

/// Run the MLIR optimizer path using the already-parsed CL options.
/// Must be called after llvm::cl::ParseCommandLineOptions().
mlir::LogicalResult runMLIRPath(llvm::StringRef inputFile,
                                llvm::StringRef outputFile,
                                mlir::DialectRegistry &registry);

} // namespace M::KGEN::Tool

#endif // KGEN_TOOLS_KGEN_OPT_MLIR_DRIVER_H
