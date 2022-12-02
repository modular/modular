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

/// Drop all uses of the current operation and nested operations and delete
/// them. This allows deletion of potentially invalid operations that use values
/// not available in its current domination tree.
inline void purgeAndErase(Operation *op) {
  op->walk([](Operation *op) { op->dropAllDefinedValueUses(); });
  op->erase();
}

} // namespace M

#endif // SUPPORT_COMPILER_OPERATIONUTILS_H
