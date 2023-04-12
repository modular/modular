//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LLVMPASSESPIPELINE_H
#define KGEN_LLVMPASSESPIPELINE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/CodeGen.h"

namespace M::KGEN {
class CompilationOptions;

/// Build a module pass pipeline for a given set of compilation options.
llvm::ModulePassManager
buildLLVMOptimizationPipeline(const CompilationOptions &options);
} // namespace M::KGEN

#endif // KGEN_LLVMPASSESPIPELINE_H
