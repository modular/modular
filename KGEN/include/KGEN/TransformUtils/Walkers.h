//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TRANSFORMUTILS_WALKERS_H
#define KGEN_TRANSFORMUTILS_WALKERS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
/// Walk the operations contained within operation in reverse, in post order.
/// That means `op` is visited after all the ops in its regions. Ops are visited
/// in reverse order in each region, starting from the last region of each op.
void reversePostOrderWalk(Operation *op,
                          function_ref<void(Operation *)> walkFn);
} // namespace M

#endif // KGEN_TRANSFORMUTILS_WALKERS_H
