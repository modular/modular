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
#define GEN_PASS_DEF_RAISEFORLOOPS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

/// This pass has to run after elaboration so that all parameter
/// expressions have elaborated with known values.
struct RaiseForLoops : impl::RaiseForLoopsBase<RaiseForLoops> {
  using RaiseForLoopsBase::RaiseForLoopsBase;

  void runOnOperation() override;

private:
  /// Map from loop to its ump Operations, i.e. BreakOp, ContinueOp.
  DenseMap<LoopOp, SmallVector<Operation *>> loopJumpOps;

  /// Parent loops.
  SmallVector<LoopOp> parentLoops;

  /// Loops to raise in program order.
  SmallVector<LoopOp> loopsToRaiseInOrder;

  /// Collect jump ops (BreakOp and ContinueOp) for loops.
  void collectJumpOps(Operation *op, StringAttr label);

  /// Walk loops in program order.
  void walkLoopsPreorder(Operation *op);

  /// Transform a simple loop that has no early exits and known iterations into
  /// a for-loop.
  void raiseForLoops(LoopOp loop);
};

struct ForLoopBoundsAndSteps {
  Value lowerBound;
  Value upperBound;
  Value step;
};

} // namespace

void RaiseForLoops::collectJumpOps(Operation *op, StringAttr label) {
  assert((isa<ContinueOp, BreakOp>(op)));

  for (LoopOp loop : llvm::reverse(parentLoops)) {
    if (isMatchingLoop(loop, label)) {
      loopJumpOps[loop].push_back(op);
      return;
    }
  }
}

void RaiseForLoops::walkLoopsPreorder(Operation *cur) {
  cur->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Associate BreakOp its target loop.
    if (auto br = dyn_cast<BreakOp>(op))
      collectJumpOps(br, br.getLabelAttr());

    // Associate ContinueOp its target loop.
    if (auto ct = dyn_cast<ContinueOp>(op))
      collectJumpOps(ct, ct.getLabelAttr());

    if (auto loop = dyn_cast<LoopOp>(op); loop && loop != cur) {
      // Recurse in nested loops.
      if (loop.getUnrollFactor().has_value())
        loopsToRaiseInOrder.push_back(loop);

      parentLoops.push_back(loop);
      walkLoopsPreorder(loop);
      parentLoops.pop_back();
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
}

template <typename OpT, typename ET>
SmallVector<OpT> getOps(ArrayRef<ET> vec) {
  SmallVector<OpT> result;
  for (ET v : vec) {
    if (auto op = dyn_cast<OpT>(v))
      result.push_back(op);
  }
  return result;
}

// If the value v is a constant get the value, otherwise return a null Value.
static Value getValueIfConstInteger(Value v, LoopOp loop) {
  Value result;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Value input = loop->getOperand(arg.getArgNumber());
    result = getValueIfConstInteger(input, loop);
  } else if (mlir::matchPattern(v, mlir::m_Constant())) {
    result = v;
  }
  return result;
}

static std::optional<ForLoopBoundsAndSteps>
inferLoopCount(LoopOp loop, ContinueOp continueOp, BreakOp breakOp) {
  // The infer logic here is assuming that for-loop's ranges are:
  // - range(n) - zero starting range with start = 0, end = n, stride = 1.
  // - range(s, e) - sequential range with start = s, end = e, stride = 1.
  // - range(s, e, st) - strided range with start = s, end = e, stride = st.
  // This is pretty limited assumption to bootstrap loop unrolling.
  // This can be improved to support more general for loops.

  // Infer loop stride from ContinueOp's input operand expression.
  Value nextIter = continueOp.getOperand(continueOp.getNumOperands() - 1);
  Value stride;
  Operation *nextIterOp = nextIter.getDefiningOp();
  if (isa<mlir::index::AddOp, mlir::index::SubOp>(nextIterOp)) {
    Value input0 = nextIterOp->getOperand(0);
    Value input1 = nextIterOp->getOperand(1);
    if (auto blockArg = dyn_cast<BlockArgument>(input0))
      stride = getValueIfConstInteger(input1, loop);
  }
  // Bail if we can't match pattern to find the stride value.
  if (!stride)
    return {};

  // Infer loop start and end from BreakOp's parent IfOp's operand expression.
  // For example:
  // %index1 = kgen.param.constant = <1>
  // %index2 = kgen.param.constant = <2>
  // %index6 = kgen.param.constant = <6>
  // %idx0 = index.constant 0
  // hlcf.loop (%arg0 = %index1 : index) {
  //    %0 = index.cmp slt(%arg0, %index9) # start = 1, end = 9
  //    hlcf.if %0 {
  //      hlcf.yield
  //    } else {
  //      hlcf.break
  //    }
  //    %1 = index.add %arg0, %index2 # stride = 2
  //    ....
  //    hlcf.continue %1 : index
  // }
  //
  IfOp ifOp = cast<IfOp>(breakOp->getParentOp());
  Value ifCond = ifOp.getOperand();
  Value start;
  Value end;
  if (auto cmp = dyn_cast<mlir::index::CmpOp>(ifCond.getDefiningOp())) {
    switch (cmp.getPred()) {
    case mlir::index::IndexCmpPredicate::SLT:
      start = getValueIfConstInteger(cmp.getLhs(), loop);
      end = getValueIfConstInteger(cmp.getRhs(), loop);
      break;
    case mlir::index::IndexCmpPredicate::SGT:
      start = getValueIfConstInteger(cmp.getRhs(), loop);
      end = getValueIfConstInteger(cmp.getLhs(), loop);
      break;
    default:
      return {};
    }
  }

  if (start && end)
    return ForLoopBoundsAndSteps{start, end, stride};

  return {};
}

void RaiseForLoops::raiseForLoops(LoopOp loop) {
  auto iter = loopJumpOps.find(loop);
  if (iter == loopJumpOps.end())
    return;

  // Only raise a loop with no early exits which should have only one BreakOp
  // and one ContinueOp.
  if (iter->second.size() != 2)
    return;

  SmallVector<BreakOp> breakOps = getOps<BreakOp, Operation *>(iter->second);
  SmallVector<ContinueOp> continueOps =
      getOps<ContinueOp, Operation *>(iter->second);

  // Only raise a loop with no early exits which should have only one BreakOp
  // and one ContinueOp.
  if (breakOps.size() != 1)
    return;

  // only raise a loop with no early exits which should have only one BreakOp
  // and one ContinueOp.
  if (continueOps.size() != 1)
    return;

  Block &body = loop.getBody().front();
  Operation *term = body.getTerminator();

  if (!isa<ContinueOp>(term))
    return;

  if (!isa<IfOp>(breakOps.front()->getParentOp()))
    return;

  std::optional<ForLoopBoundsAndSteps> loopInfo =
      inferLoopCount(loop, continueOps.front(), breakOps.front());

  if (!loopInfo.has_value())
    return;

  mlir::IRRewriter rewriter{OpBuilder(loop)};
  IRMapping map;
  ForOp forOp = rewriter.create<HLCF::ForOp>(
      loop->getLoc(), loop->getResultTypes(), loopInfo->lowerBound,
      loopInfo->upperBound, loopInfo->step, loop.getOperands(),
      loop.getLabelAttr(), loop.getUnrollFactorAttr());

  Block *block = rewriter.createBlock(&forOp.getBody());

  for (BlockArgument arg : body.getArguments()) {
    rewriter.replaceAllUsesWith(
        arg, block->addArgument(arg.getType(), arg.getLoc()));
  }

  Operation *prevOp = nullptr;
  for (Operation &op : llvm::make_early_inc_range(body.getOperations())) {
    if (&op == breakOps.front()->getParentOp() && isa<IfOp>(op)) {
      // Don't move the parent IfOp of the break to the ForOp body.
      continue;
    }

    // Move op to the ForOp body.
    if (prevOp == nullptr) {
      op.moveBefore(block, block->begin());
    } else {
      op.moveAfter(prevOp);
      if (auto c = dyn_cast<ContinueOp>(op)) {
        rewriter.setInsertionPointAfter(&op);
        rewriter.create<HLCF::ForYieldOp>(op.getLoc(), c.getOperands(),
                                          c.getLabelAttr());
        c->dropAllReferences();
        rewriter.eraseOp(c);
      }
    }
    prevOp = &op;
  }

  loop->replaceAllUsesWith(forOp.getResults());

  // Erase the original loop.
  rewriter.eraseOp(loop);
}

void RaiseForLoops::runOnOperation() {
  loopJumpOps.clear();
  loopsToRaiseInOrder.clear();
  parentLoops.clear();

  walkLoopsPreorder(getOperation());
  // raise for-loops from inner to outer
  for (LoopOp loop : llvm::reverse(loopsToRaiseInOrder)) {
    raiseForLoops(loop);
  }
}

