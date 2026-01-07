//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// LLVM IR Downgrade Pass - Transform LLVM IR for backend compilation
// that takes older version of LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_LLVMIR_TRANSFORMS_LLVMIRDOWNGRADEPASS_H
#define KGEN_COMPILER_LLVMIR_TRANSFORMS_LLVMIRDOWNGRADEPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
class ModulePass;
} // namespace llvm

namespace M::KGEN {

/// New pass manager pass that transforms LLVM IR for backend compilation
// that takes older version of LLVM IR.
///
/// This pass:
/// - Transforms llvm.lifetime related intrinsics
///
class LLVMIRDowngradePass : public llvm::PassInfoMixin<LLVMIRDowngradePass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMIR_TRANSFORMS_LLVMIRDOWNGRADEPASS_H
