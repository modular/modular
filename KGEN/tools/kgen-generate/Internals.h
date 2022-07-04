//===- Internals.h --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef INTERNALS_H
#define INTERNALS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
/// Generate kernels in the specified module, incorporating implementation logic
/// from the specified library.  On error, diagnostics are emitted and the
/// primary file isn't completely lowered.
LogicalResult generateKernels(ModuleOp primary, ModuleOp library);

/// This is like `Operation::clone`, but instead of just keeping track of the
/// block and value mapping for the copy, it also keeps track of the
/// operation<->operation mapping.  This matters because not all operations have
/// results.
Operation *cloneOperation(Operation *original, BlockAndValueMapping &mapper,
                          DenseMap<Operation *, Operation *> &operationMap);

} // namespace M

#endif // INTERNALS_H
