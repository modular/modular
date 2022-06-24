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
/// from the specified library.
LogicalResult generateKernels(ModuleOp module, ModuleOp library);
} // namespace M

#endif // INTERNALS_H
