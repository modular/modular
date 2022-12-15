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
} // namespace M

#endif // SUPPORT_COMPILER_OPERATIONUTILS_H
