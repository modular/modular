//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Interface for the kgen-opt LLVM IR processing path.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_KGEN_OPT_LLVM_DRIVER_H
#define KGEN_TOOLS_KGEN_OPT_LLVM_DRIVER_H

#include "llvm/ADT/StringRef.h"

namespace M::KGEN::Tool {

/// Register all LLVM-path CL options (-passes, -O0/-O3, -S, -mtriple, …).
/// Must be called before llvm::cl::ParseCommandLineOptions().
void registerLLVMPathCLOptions();

/// Run the LLVM IR optimizer path using the already-parsed CL options.
/// Must be called after llvm::cl::ParseCommandLineOptions().
/// Returns a process exit code (0 = success).
int runLLVMPath(llvm::StringRef inputFile, llvm::StringRef outputFile);

} // namespace M::KGEN::Tool

#endif // KGEN_TOOLS_KGEN_OPT_LLVM_DRIVER_H
