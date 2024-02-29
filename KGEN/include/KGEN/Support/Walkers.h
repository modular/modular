//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_WALKERS_H
#define KGEN_SUPPORT_WALKERS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Operation.h"

namespace M {
/// Walk the operations contained within operation in reverse, in post order.
/// That means `op` is visited after all the ops in its regions. Ops are visited
/// in reverse order in each region, starting from the last region of each op.
inline void reversePostOrderWalk(Operation *op,
                                 function_ref<void(Operation *)> walkFn) {
  for (Region &region : llvm::reverse(op->getRegions())) {
    // There shouldn't be more than one block here.
    assert(region.getBlocks().size() <= 1 && "unexpected CFG");
    if (!region.hasOneBlock())
      continue;
    Block &block = region.front();
    // Ops can get deleted, so make sure to early inc.
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(block)))
      reversePostOrderWalk(&op, walkFn);
  }
  walkFn(op);
}
} // namespace M

#endif // KGEN_SUPPORT_WALKERS_H
