//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace POP;

namespace M::KGEN {
#define GEN_PASS_DEF_MEM2REG
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class Mem2RegPass : public M::KGEN::impl::Mem2RegBase<Mem2RegPass> {
public:
  void runOnOperation() override;

private:
  /// Check whether the operation has only single block regions.
  bool allSingleBlock(Operation *op) {
    for (Region &region : op->getRegions()) {
      if (!region.empty() && !llvm::hasSingleElement(region))
        return false;
      for (Operation &op : region.getOps())
        if (op.getNumRegions() && !allSingleBlock(&op))
          return false;
    }
    return true;
  }

  /// Promote a single alloc given its users in order.
  void promoteAlloc(StackAllocationOp alloc, ArrayRef<Operation *> users);
};
} // namespace

/// We can promote a stack allocation value to a register if:
/// - it allocates one element and uses the default alignment
/// - its only uses are loads anywhere and stores within its own scope
///
/// `Operation::getUsers` does not return the users in any particular order, so
/// we have to walk the IR in order to get the users in sequence.
void Mem2RegPass::runOnOperation() {
  // Require that this operation has only single-block regions.
  if (!allSingleBlock(getOperation())) {
    getOperation()->emitError("'ssa-formation' can only be run on operations "
                              "with all single block regions");
    return signalPassFailure();
  }

  DenseMap<StackAllocationOp, std::vector<Operation *>> users;
  getOperation()->walk([&](Operation *op) {
    if (auto alloc = dyn_cast<StackAllocationOp>(op)) {
      // Initialize the users for this alloc if it is valid for SSA formation.
      auto count = dyn_cast<IntegerAttr>(alloc.getCount());
      if (count && count.getInt() == 1 && !alloc.getAlignmentAttr())
        users.insert({alloc, {}});
      return;
    }
    if (auto load = dyn_cast<LoadOp>(op)) {
      // If the load is a user of an alloc that is still valid for SSA
      // formation, add it as a user.
      if (auto alloc = load.getPtr().getDefiningOp<StackAllocationOp>())
        if (auto it = users.find(alloc); it != users.end())
          it->second.push_back(load);
      return;
    }
    if (auto store = dyn_cast<StoreOp>(op)) {
      // If the store is a user of an alloc that is still valid for SSA
      // formation and the store is in the same region as the alloc, then add it
      // as a user. If it is in another scope, however, invalidate the alloc for
      // SSA formation.
      if (auto alloc = store.getPtr().getDefiningOp<StackAllocationOp>()) {
        if (auto it = users.find(alloc); it != users.end()) {
          if (store->getBlock() == alloc->getBlock())
            it->second.push_back(store);
          else
            users.erase(it);
        }
      }
      // Using an alloc as the store value invalidates it.
      if (auto alloc = store.getArg().getDefiningOp<StackAllocationOp>())
        users.erase(alloc);
      return;
    }
    // Any other use of a stack allocation invalidates it.
    for (Value operand : op->getOperands())
      if (auto alloc = operand.getDefiningOp<StackAllocationOp>())
        users.erase(alloc);
  });

  // Now go through the valid allocs and their users and rewrite them.
  for (auto &[alloc, users] : users)
    promoteAlloc(alloc, users);
}

void Mem2RegPass::promoteAlloc(StackAllocationOp alloc,
                               ArrayRef<Operation *> users) {
  // Track the current value of the memory.
  Value curVal;
  for (Operation *user : users) {
    if (auto load = dyn_cast<LoadOp>(user)) {
      // We can't elide the load if it is loading uninitialized memory.
      if (!curVal) {
        load.emitWarning("load of uninitialized memory")
                .attachNote(alloc.getLoc())
            << "memory allocated here";
        return;
      }
      // Replace the load with the current value.
      load.replaceAllUsesWith(curVal);
      load->erase();
      ++numLoadsElided;
    } else {
      // The user must be a store. Save the new value.
      auto store = cast<StoreOp>(user);
      curVal = store.getArg();
      store->erase();
      ++numStoresElided;
    }
  }
  alloc->erase();
  ++numAllocsElided;
}
