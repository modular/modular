//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_OPERATIONUTILS_H
#define SUPPORT_COMPILER_OPERATIONUTILS_H

#include "Support/Compiler/MLIRToString.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"

namespace M {
/// Generate a unique flat symbol name with respect to the provided symbol table
/// given a base name. This method is useful if one wants a unique symbol name
/// before creating a function. The caller should provide a base ID to re-use,
/// which is incremented until a unique name is found.
std::string getUniqueSymbolName(std::string baseName, SymbolTable &symtab,
                                unsigned &counter);
/// Generate a unique flat symbol name with respect to the provided symbol table
/// given a base name. This version starts the counter from 0.
std::string getUniqueSymbolName(std::string baseName, SymbolTable &symtab);

/// Flatten the given symbol reference into a single string, concatenating the
/// namespaces and the symbol name. For example, given a symbol
/// `@foo::@bar::@baz`, this function returns `foo::bar::baz`.
std::string getFlattenedSymbolName(SymbolRefAttr symbol);

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
