//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_LLVMIR_TRANSFORMS_SETFUNCTIONATTRIBUTES_H
#define KGEN_COMPILER_LLVMIR_TRANSFORMS_SETFUNCTIONATTRIBUTES_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN {

/// Pass to set some function attributes that are needed for compilation.
class SetFunctionAttributes
    : public llvm::PassInfoMixin<SetFunctionAttributes> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

  static llvm::StringRef name() { return "KGEN::SetFunctionAttributes"; }

private:
  void
  runImpl(llvm::Module &M,
          const llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &options);
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMIR_TRANSFORMS_SETFUNCTIONATTRIBUTES_H
