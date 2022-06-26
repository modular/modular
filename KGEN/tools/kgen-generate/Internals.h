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
} // namespace M

#endif // INTERNALS_H
