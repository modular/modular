//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_LLVMIR_TRANSFORMS_POINTERREWRITER_H
#define KGEN_COMPILER_LLVMIR_TRANSFORMS_POINTERREWRITER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class Value;
class Module;
class TypedPointerType;
} // namespace llvm

namespace M {
namespace KGEN {

/// Pass to rewrite opaque pointers to typed pointers that constructs a map
/// between opaque pointer and it's intended type.
class PointerRewriter : public llvm::PassInfoMixin<PointerRewriter> {
public:
  using PointerTypeMap =
      llvm::DenseMap<const llvm::Value *, llvm::TypedPointerType *>;

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

  static llvm::StringRef name() { return "PointerRewriter"; }

  static PointerTypeMap buildPointerMap(const llvm::Module &M);

private:
  bool runImpl(llvm::Module &M);
  bool cleanupTypedPointerMetadata(llvm::Module &M);
};

} // namespace KGEN
} // namespace M

#endif // KGEN_COMPILER_LLVMIR_TRANSFORMS_POINTERREWRITER_H
