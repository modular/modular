//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_OPERATIONUTILS_H
#define SUPPORT_COMPILER_OPERATIONUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"

namespace M {
/// Given an operation, determine whether any nested operations use values
/// captured from above. Store those captures in the `captures` pointer if it's
/// provided.
bool operationIsIsolatedFromAbove(Operation *op,
                                  llvm::SetVector<Value> *captures = nullptr,
                                  bool allowIsolated = false);

/// Generate a unique flat symbol name with respect to the provided symbol table
/// given a base name. This method is useful if one wants a unique symbol name
/// before creating a function. The caller should provide a base ID to re-use,
/// which is incremented until a unique name is found.
std::string getUniqueSymbolName(std::string baseName, SymbolTable &symtab,
                                unsigned &counter);

/// Check if ident is a valid C identifier: it contains only the
/// characters in the set [0-9a-zA-Z_] and it cannot start with a '_'.
bool isCIdentifier(StringRef ident);

/// Return the nearest parent operation of the block of the given kind.
template <typename OpT>
OpT getBlockParentOfType(Block *block) {
  if (auto op = dyn_cast<OpT>(block->getParentOp()))
    return op;
  return block->getParentOp()->getParentOfType<OpT>();
}
} // namespace M

#endif // SUPPORT_COMPILER_OPERATIONUTILS_H
