//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LLVMPASSESPIPELINE_H
#define KGEN_LLVMPASSESPIPELINE_H
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CodeGen.h"

// Build a module pass pipeline for a given optimization level (only O0 and O3
// are supported).
llvm::ModulePassManager buildPipeline(llvm::CodeGenOpt::Level Level);

#endif // KGEN_LLVMPASSESPIPELINE_H
