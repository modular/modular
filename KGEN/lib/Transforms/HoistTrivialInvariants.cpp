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
    : M::KGEN::impl::HoistTrivialInvariantsBase<HoistTrivialInvariants> {
  void runOnOperation() override;
};

/// Hoist invariant operations to the earlest legal point we can within the
/// function. Either to the start if they use only input arguments or to the
/// producer of whichever operand is dominated by all other operands.
static void moveInvariants(KGEN::FuncOp func, Operation *opWithRegion,
                           iterator_range<Region::OpIterator> range) {
  Block &entryBlock = func.getBodyRegion().front();

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
      Operation *parent = operand.getDefiningOp();

      // Only allow hoisting ops with block argument operands if those operands
      // are function level.
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        if (blockArg.getOwner() != &entryBlock) {
          safe = false;
          break;
        }
        continue;
      }

      if (parent->getParentOp() == opWithRegion) {
        safe = false;
        break;
      }
    }

    if (!safe)
      continue;

    // The operand which is dominated by all others.
    Operation *leastDominatingOperand = nullptr;
    bool allOperandsAreBlocks = true;

    mlir::DominanceInfo domTree;

    // Traverse again to avoid touching dom info on region variant ops.
    for (Value operand : op.getOperands()) {
      Operation *parent = operand.getDefiningOp();

      // Don't need any domanince info for block args.
      if (isa<BlockArgument>(operand))
        continue;

      allOperandsAreBlocks = false;

      // Find the insertion point.
      if (!leastDominatingOperand) {
        leastDominatingOperand = parent;
      } else {
        if (domTree.dominates(leastDominatingOperand, parent))
          leastDominatingOperand = parent;
      }
    }

    // If we are moving an operation that only uses the function block args we
    // hoist to the start of the function. Otherwise hoist to the earliest legal
    // point.
    if (allOperandsAreBlocks)
      op.moveAfter(&entryBlock, entryBlock.begin());
    else if (leastDominatingOperand)
      op.moveAfter(leastDominatingOperand);
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
