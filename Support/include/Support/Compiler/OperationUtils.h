//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_OPERATIONUTILS_H
#define SUPPORT_COMPILER_OPERATIONUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Operation.h"

namespace M {
/// This is like `Operation::clone`, but instead of just keeping track of the
/// block and value mapping for the copy, it also keeps track of the
/// operation<->operation mapping.  This matters because not all operations have
/// results.
Operation *cloneOperation(Operation *original, BlockAndValueMapping &mapper,
                          DenseMap<Operation *, Operation *> &operationMap);

/// Given an operation, determine whether any nested operations use values
/// captured from above.
bool operationIsIsolatedFromAbove(Operation *op);

/// Generate a unique flat symbol name with respect to the provided symbol table
/// given a base name. This method is useful if one wants a unique symbol name
/// before creating a function. The caller should provide a base ID to re-use,
/// which is incremented until a unique name is found.
std::string getUniqueSymbolName(std::string baseName, SymbolTable &symtab,
                                unsigned &counter);
/// Generate a valid C identifier from the input identifier. A valid C
/// identifier contains only the characters in the set [0-9a-zA-Z_].
std::string makeCIdentifier(StringRef ident);
} // namespace M

#endif // SUPPORT_COMPILER_OPERATIONUTILS_H
