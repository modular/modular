//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_TRANSFORMS_CONTROLFLOWUTILS_H
#define KGEN_LIB_TRANSFORMS_CONTROLFLOWUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN {
/// Return true if the user of the provided operation is outside the contiguous
/// CFG in which the operation lives. A contiguous CFG is defined as a region
/// subtree where all region operations implement an HLCF interface. Any other
/// operation is assumed to break the CFG, such as inline closures.
bool userCrossesFunctionCFG(Operation *op, Operation *user);
} // namespace M::KGEN

#endif // KGEN_LIB_TRANSFORMS_CONTROLFLOWUTILS_H
