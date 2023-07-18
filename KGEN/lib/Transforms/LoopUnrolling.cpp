//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/HLCFDialect/HLCFUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace HLCF;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOOPUNROLLING
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
/// Unroll For-Loops with unrollFactor attributes:
/// - fully unroll a for-loop.
/// - unroll a for-loop with an unroll factor of a constant value .
/// This pass has to run after elaboration so that all parameter
/// expressions have elaborated with known values.
struct LoopUnrolling : impl::LoopUnrollingBase<LoopUnrolling> {
  using LoopUnrollingBase::LoopUnrollingBase;

  void runOnOperation() override;

private:
  SmallVector<ForOp> parentLoops;

  /// Loops to unroll in program order.
  SmallVector<ForOp> loopsToUnrollInOrder;

  /// Walk loops in program order.
  void walkLoopsPreorder(Operation *op);

  /// Fully unroll a simple loop that has no early exits and known iterations.
  LogicalResult fullUnrollForLoop(ForOp loop);
};
} // namespace

void LoopUnrolling::walkLoopsPreorder(Operation *cur) {
  cur->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto loop = dyn_cast<ForOp>(op); loop && loop != cur) {
      // Recurse in nested loops.
      loopsToUnrollInOrder.push_back(loop);

      parentLoops.push_back(loop);
      walkLoopsPreorder(loop);
      parentLoops.pop_back();
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
}

LogicalResult LoopUnrolling::fullUnrollForLoop(ForOp loop) {
  std::optional<int64_t> count = loop.getTripCount();
  if (!count)
    return failure();

  mlir::IRRewriter rewriter{OpBuilder(loop)};

  Region &scopeBody = loop->getParentOp()->getRegion(0);
  Block &body = loop.getBody().front();

  ForYieldOp newForYield;
  SmallVector<Value> retValues;
  retValues = loop.getIterArgs();

  for (int64_t i = 0; i < count; ++i) {
    IRMapping map;
    Block *block = rewriter.createBlock(&scopeBody);

    if (i + 1 == count) {
      // Last iteration, move ops instead of cloning.
      for (BlockArgument arg : body.getArguments())
        rewriter.replaceAllUsesWith(
            arg, block->addArgument(arg.getType(), arg.getLoc()));

      Operation *prevOp = nullptr;
      for (Operation &op : llvm::make_early_inc_range(body.getOperations())) {
        if (auto y = dyn_cast<ForYieldOp>(op)) {
          // Don't move last ForYieldOp.
          newForYield = y;
          continue;
        }

        // Move ops to the last block of inline.
        if (!prevOp)
          op.moveBefore(block, block->begin());
        else
          op.moveAfter(prevOp);

        prevOp = &op;
      }

      // Add unrolled block before the loop.
      rewriter.inlineBlockBefore(block, loop, retValues);

      // Get result value of the loop.
      retValues = newForYield.getOperands();
      break;
    }

    for (BlockArgument arg : body.getArguments())
      map.map(arg, block->addArgument(arg.getType(), arg.getLoc()));

    for (Operation &op : body.getOperations()) {
      auto newOp = rewriter.clone(op, map);
      if (auto y = dyn_cast<ForYieldOp>(newOp))
        newForYield = y;
    }

    // Add unrolled block before the loop.
    rewriter.inlineBlockBefore(block, loop, retValues);

    // Update next iteration's inputs
    retValues = newForYield.getOperands();

    // Erase ForYieldOp.
    rewriter.eraseOp(newForYield);
  }

  // Replace the loop return value, which are first group of operands of
  // ForYieldOp.
  loop.replaceAllUsesWith(llvm::drop_begin(
      llvm::drop_end(retValues, retValues.size() - loop.getNumResults() - 1)));

  // Erase the original loop.
  rewriter.eraseOp(loop);

  return success();
}

void LoopUnrolling::runOnOperation() {
  parentLoops.clear();
  loopsToUnrollInOrder.clear();

  walkLoopsPreorder(getOperation());
  // unroll loops from inner to outer
  for (auto loop : llvm::reverse(loopsToUnrollInOrder)) {
    if (loop.isFullUnroll() || (loop.getTripCount() == 1)) {
      // Fully unroll if loop is decorated or has single iteration.
      if (succeeded(fullUnrollForLoop(loop)))
        continue;
      // TODO: unroll with a factor based on cost model if a for loop decorated
      // with fully unroll is not a loop that has no early exits.
    } else {
      // TODO: unroll loops with decorator of an unroll factor
    }
  }
}
