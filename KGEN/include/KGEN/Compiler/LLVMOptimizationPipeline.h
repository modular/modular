//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_LLVMPASSESPIPELINE_H
#define KGEN_COMPILER_LLVMPASSESPIPELINE_H

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"

namespace M::KGEN {

class CompilationOptions;

/// Build a module pass pipeline for a given set of compilation options.
llvm::ModulePassManager
buildLLVMOptimizationPipeline(llvm::PassBuilder &passBuilder,
                              const CompilationOptions &options);

/// Add LLVMIRDowngradePass to the pass manager.
void addLLVMIRDowngradePass(llvm::ModulePassManager &mpm);

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMPASSESPIPELINE_H
