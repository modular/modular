//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/HLCFDialect/HLCFUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace HLCF;

namespace M::KGEN {
#define GEN_PASS_DEF_SIMPLIFYCF
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct SimplifyCF : M::KGEN::impl::SimplifyCFBase<SimplifyCF> {
  void runOnOperation() override;

private:
  void tryRemovingLoop(LoopOp loop);
  LoopOp getOrFindTargetLoop(Operation *op, StringAttr label);
  void walkLoopsPreorder(Operation *cur);

  /// A map for looking up the target loop for the op (break or continue).
  DenseMap<Operation *, LoopOp> targetLoopMap;

  /// Function loops in the pre-order.
  SmallVector<LoopOp> loopsInOrder;

  /// Outer loops stack for the walk.
  SmallVector<LoopOp> parentLoops;

  /// Number of break or continue ops that lead to the current loop (breaks and
  /// continues in inner loops don't count).
  DenseMap<LoopOp, int> jumpsCount;
};
} // namespace

LoopOp SimplifyCF::getOrFindTargetLoop(Operation *op, StringAttr label) {
  assert((isa<ContinueOp, BreakOp>(op)));
  auto it = targetLoopMap.find(op);
  if (it != targetLoopMap.end())
    return it->second;

  assert(!parentLoops.empty());
  for (auto loop : llvm::reverse(parentLoops)) {
    if (isMatchingLoop(loop, label)) {
      ++jumpsCount[loop];
      targetLoopMap[op] = loop;
      return loop;
    }
  }
  return nullptr;
}

/// If the loop body ends with BreakOp and there are no other break or continue
/// ops in the body, the loop can be removed.  Such loops are often generated as
/// inlining by-product, where breaks are used to represent returns from the
/// inlined function.
///
/// Before:                    After:
/// {                          {
///   ...                        ...
///   %x = hlcf.loop {
///      %a = A                  %x = A
///      hlcf.break %a
///   }
///   C                          C
/// }                          }
void SimplifyCF::tryRemovingLoop(LoopOp loop) {
  Block &body = loop.getBody().front();

  // If the loop has more than just one break or continue in it, we can't remove
  // it.
  if (jumpsCount.at(loop) > 1)
    return;

  // Check that the body ends with a break or return.
  Operation *term = body.getTerminator();
  if (!isa<BreakOp, KGEN::ReturnOp>(term))
    return;

  // If the loop body ends with return, but jumpsCount is 1, it means that there
  // is a break or continue somewhere inside the loop body - we can't deal with
  // that.
  if (isa<KGEN::ReturnOp>(term) && jumpsCount.at(loop) != 0)
    return;

  // All the checks passed, the loop now can be removed!
  mlir::IRRewriter rewriter{OpBuilder(loop)};

  rewriter.inlineBlockBefore(&body, loop);
  loop.replaceAllUsesWith(term->getOperands());

  rewriter.eraseOp(term);
  rewriter.eraseOp(loop);
}

void SimplifyCF::walkLoopsPreorder(Operation *cur) {
  cur->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto c = dyn_cast<ContinueOp>(op))
      getOrFindTargetLoop(op, c.getLabelAttr());

    if (auto br = dyn_cast<BreakOp>(op))
      getOrFindTargetLoop(op, br.getLabelAttr());

    // keep walking until we see another loop, then we recurse
    if (auto loop = dyn_cast<LoopOp>(op); loop && loop != cur) {
      loopsInOrder.push_back(loop);
      parentLoops.push_back(loop);
      walkLoopsPreorder(op);
      parentLoops.pop_back();
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
}

void SimplifyCF::runOnOperation() {
  targetLoopMap.clear();
  loopsInOrder.clear();
  parentLoops.clear();
  jumpsCount.clear();

  // Walk over the functions and count how many jumps (breaks or continues) each
  // loop has. Note that breaks and continues targeting inner loops do not count
  // as jumps in this context - we only care about control flow transfers that
  // move us out of this loop.
  walkLoopsPreorder(getOperation());

  // If a loop has just one jump, we can try removing it.
  for (LoopOp loop : loopsInOrder)
    if (jumpsCount[loop] <= 1)
      tryRemovingLoop(loop);
}
