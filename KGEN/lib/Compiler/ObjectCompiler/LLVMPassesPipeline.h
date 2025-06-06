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
struct MCInfo;

/// Build a module pass pipeline for a given set of compilation options.
llvm::ModulePassManager
buildLLVMOptimizationPipeline(llvm::PassBuilder &passBuilder,
                              const CompilationOptions &options);

bool addPassesToEmitFile(CompilationOptions &options,
                         llvm::TargetMachine &targetMachine,
                         llvm::legacy::PassManagerBase &pm,
                         llvm::raw_pwrite_stream &out,
                         llvm::raw_pwrite_stream *dwoOut,
                         llvm::CodeGenFileType fileType, bool disableVerify,
                         llvm::MachineModuleInfoWrapperPass *mmiwp);

/// Build a pipeline that does machine specific codegen but stops before
/// AsmPrint.
bool addPassesToEmitMC(CompilationOptions &options,
                       llvm::TargetMachine &targetMachine,
                       llvm::legacy::PassManagerBase &pm,
                       llvm::raw_pwrite_stream &out, bool disableVerify,
                       llvm::MachineModuleInfoWrapperPass *mmiwp,
                       unsigned numFnBase);

/// Build a pipeline that does AsmPrint only.
bool addPassesToAsmPrint(CompilationOptions &options,
                         llvm::TargetMachine &targetMachine,
                         llvm::legacy::PassManagerBase &pm,
                         llvm::raw_pwrite_stream &out,
                         llvm::CodeGenFileType fileType, bool disableVerify,
                         llvm::MachineModuleInfoWrapperPass *mmiwp,
                         llvm::SmallVectorImpl<MCInfo *> &mcInfos);

} // namespace M::KGEN

#endif // KGEN_LLVMPASSESPIPELINE_H
