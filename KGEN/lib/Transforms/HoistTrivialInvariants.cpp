//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/Dominance.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_HOISTTRIVIALINVARIANTS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct HoistTrivialInvariants
    : impl::HoistTrivialInvariantsBase<HoistTrivialInvariants> {
  void runOnOperation() override;
};

/// Hoist invariant operations to the earlest legal point we can within the
/// function. Either to the start if they use only input arguments or to the
/// producer of whichever operand is dominated by all other operands.
static void moveInvariants(FuncOp func, Operation *opWithRegion,
                           iterator_range<Region::OpIterator> range) {
  // Move the invariants.
  for (Operation &op : llvm::make_early_inc_range(range)) {
    // This pass only will hoist
    if (!isPure(&op))
      continue;

    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    // A pure operation is invariant if all of its operands are invariant.
    // In basic invariant code motion we just check if is created within this
    // loop / if, if so we don't move it. Otherwise we assume it is safe to move
    // and leave LLVM to decide whether or not it is more performant for it to
    // be hoisted back in.
    bool safe = true;
    for (Value operand : op.getOperands()) {
      if (operand.getParentRegion()->getParentOp() == opWithRegion) {
        safe = false;
        break;
      }
    }

    if (!safe)
      continue;

    // The operand which is dominated by all others.
    PointerUnion<Operation *, Region *> leastDominatingOperand = nullptr;
    mlir::DominanceInfo domTree;

    // Traverse again to avoid touching dom info on region variant ops.
    for (Value operand : op.getOperands()) {
      if (Operation *parent = operand.getDefiningOp()) {
        if (!leastDominatingOperand) {
          leastDominatingOperand = parent;
        } else if (auto *op = leastDominatingOperand.dyn_cast<Operation *>()) {
          if (domTree.dominates(op, parent))
            leastDominatingOperand = parent;
        } else if (leastDominatingOperand.get<Region *>()->isAncestor(
                       parent->getParentRegion())) {
          leastDominatingOperand = parent;
        }
      } else {
        Region *region = operand.getParentRegion();
        if (!leastDominatingOperand) {
          leastDominatingOperand = region;
        } else if (auto *op = leastDominatingOperand.dyn_cast<Operation *>()) {
          if (op->getParentRegion()->isProperAncestor(region))
            leastDominatingOperand = region;
        } else if (leastDominatingOperand.get<Region *>()->isProperAncestor(
                       region)) {
          leastDominatingOperand = region;
        }
      }
    }

    // Hoist to the earliest legal point. This is the start of the region if the
    // least dominating operand is a block argument. Otherwise, move to the
    // start of the region.
    if (!leastDominatingOperand)
      op.moveBefore(func.getBody(), func.getBody()->begin());
    else if (auto *region = leastDominatingOperand.dyn_cast<Region *>())
      op.moveBefore(&region->front(), region->front().begin());
    else
      op.moveAfter(leastDominatingOperand.get<Operation *>());
  }
}

} // namespace

void HoistTrivialInvariants::runOnOperation() {
  FuncOp func = getOperation();

  func.walk([&](Operation *opWithRegion) {
    // We maintain a small list of operations which we are allowed to hoist
    // invariants from.
    if (auto loop = dyn_cast<HLCF::LoopOp>(opWithRegion))
      moveInvariants(func, loop, loop.getOps());

    // We hoist from both branches of the if regardless of the condition with
    // the guarantee that these ops have no side effects and LLVM is free to
    // move them back if that is more optimal.
    if (auto ifOp = dyn_cast<HLCF::IfOp>(opWithRegion)) {
      moveInvariants(func, ifOp, ifOp.getThenRegion().getOps());
      moveInvariants(func, ifOp, ifOp.getElseRegion().getOps());
    }

    if (auto tryOp = dyn_cast<LIT::TryOp>(opWithRegion)) {
      moveInvariants(func, tryOp, tryOp.getTryRegion().getOps());
      moveInvariants(func, tryOp, tryOp.getExceptRegion().getOps());
      moveInvariants(func, tryOp, tryOp.getElseRegion().getOps());
    }
  });
}
