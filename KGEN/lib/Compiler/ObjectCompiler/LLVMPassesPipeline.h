//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LLVMPASSESPIPELINE_H
#define KGEN_LLVMPASSESPIPELINE_H

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"

namespace M::KGEN {
class CompilationOptions;

/// Build a module pass pipeline for a given set of compilation options.
llvm::ModulePassManager
buildLLVMOptimizationPipeline(const CompilationOptions &options);

bool addPassesToEmitFile(CompilationOptions &options,
                         llvm::LLVMTargetMachine &targetMachine,
                         llvm::legacy::PassManagerBase &pm,
                         llvm::raw_pwrite_stream &out,
                         llvm::raw_pwrite_stream *dwoOut,
                         llvm::CodeGenFileType fileType, bool disableVerify,
                         llvm::MachineModuleInfoWrapperPass *mmiwp);

} // namespace M::KGEN

#endif // KGEN_LLVMPASSESPIPELINE_H
