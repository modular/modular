//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

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
// TODO: raise any LoopOp with or without an unroll factor if possible.
struct RaiseForLoops : impl::RaiseForLoopsBase<RaiseForLoops> {
  using RaiseForLoopsBase::RaiseForLoopsBase;
  explicit RaiseForLoops(const RaiseForLoopsOptions &options = {})
      : RaiseForLoopsBase(options) {}

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
  LogicalResult raiseForLoops(LoopOp loop, InFlightDiagnostic &diag);
};

struct ForLoopBoundsAndSteps {
  Value lowerBound;
  Value upperBound;
  Value step;
  // Position number in the BlockArgument list where the induction variable is.
  int64_t inductionVarArgNumber;
  HLCF::ForLoopBoundCmpPredicate cmpPredicate;
  HLCF::ForLoopIndVarCompute indVarCompute;
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

/// Return whether `v` is a const value.
static bool isConstValue(Value v) {
  return mlir::matchPattern(v, mlir::m_Constant());
}

/// If `v` is a constant at the start of the loop, return that const value.
/// If `v` is passed in as the initial operand of the loop, argNumber is filled
/// with that operand number.
static Value getValueIfConstAtLoopStart(Value v,
                                        std::optional<int64_t> &argNumber,
                                        LoopOp currLoop) {
  Value result;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    // Return the initial value of a block argument if it is constant.
    argNumber = arg.getArgNumber();
    if (arg.getParentBlock()->getParentOp() == currLoop) {
      Value input = currLoop->getOperand(arg.getArgNumber());
      if (isConstValue(input))
        result = input;
    }
  } else if (isConstValue(v)) {
    // If the value v is a constant get the value.
    result = v;
  }
  // Otherwise return a null Value.
  return result;
}

/// If `v` is a constant in every iteration of the loop, return that const
/// value. Requires `continueOp` to be the only continue in the loop.
static Value getValueIfConstEveryIteration(Value v, LoopOp loop,
                                           ContinueOp continueOp) {
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (arg.getParentBlock()->getParentOp() == loop) {
      // Make sure the initial input is constant.
      Value initialConst = loop.getOperand(arg.getArgNumber());
      if (!isConstValue(initialConst))
        return {};

      // Make sure continue is using the same value.
      unsigned argNumber = arg.getArgNumber();
      if (continueOp->getOperand(argNumber) != arg)
        return {};

      return initialConst;
    }
  } else if (isConstValue(v)) {
    return v;
  }
  return {};
}

// Match CmpOp with specific predicateTypes
class CmpOpMatcher {
public:
  CmpOpMatcher(const SmallVector<mlir::index::IndexCmpPredicate> &predTypes)
      : predicateTypes(predTypes) {}

  bool match(Operation *op) {
    if (auto c = dyn_cast<mlir::index::CmpOp>(op))
      if (llvm::is_contained(predicateTypes, c.getPred()))
        cmpOp = c;
    return cmpOp;
  }
  mlir::index::CmpOp cmpOp;

private:
  SmallVector<mlir::index::IndexCmpPredicate> predicateTypes;
};

static HLCF::ForLoopBoundCmpPredicate
invertCmpPred(HLCF::ForLoopBoundCmpPredicate pred) {
  switch (pred) {
  case HLCF::ForLoopBoundCmpPredicate::SGT:
    return HLCF::ForLoopBoundCmpPredicate::SLE;
  case HLCF::ForLoopBoundCmpPredicate::SLT:
    return HLCF::ForLoopBoundCmpPredicate::SGE;
  case HLCF::ForLoopBoundCmpPredicate::SGE:
    return HLCF::ForLoopBoundCmpPredicate::SLT;
  case HLCF::ForLoopBoundCmpPredicate::SLE:
    return HLCF::ForLoopBoundCmpPredicate::SGT;
  }
  llvm_unreachable("invalid cmp predicate");
}

static std::optional<ForLoopBoundsAndSteps>
inferLoopCount(LoopOp loop, ContinueOp continueOp, BreakOp breakOp) {
  // The infer logic here is assuming that for-loop's ranges are:
  // - range(n) - zero starting range with start = 0, end = n, stride = 1.
  // - range(s, e) - sequential range with start = s, end = e, stride = 1.
  // - range(s, e, st) - strided range with start = s, end = e, stride = st.
  // This is pretty limited assumption to bootstrap loop unrolling.
  // This can be improved to support more general for loops.

  // Infer loop start and end from BreakOp's parent IfOp's operand expression.
  // For example:
  // %index1 = kgen.param.constant = <1>
  // %index2 = kgen.param.constant = <2>
  // %index6 = kgen.param.constant = <6>
  // %idx0 = index.constant 0
  // %0 = hlcf.loop (%arg0 = %index1 : index, %arg1 = %index2: index, %arg2 =
  // %index6) {
  //    %1 = index.cmp slt(%arg0, %index9) # start = 1, end = 9
  //    hlcf.if %1 {
  //      hlcf.yield
  //    } else {
  //      hlcf.break %arg1
  //    }
  //    %1 = index.add %arg0, %index2 # stride = 2
  //    ....
  //    hlcf.continue %1 : index
  // }
  //
  // From hlcf.if %1, we can infer that %arg0 is the induction variable
  // (inductionVarArgNumber = 0)
  // From hlcf.break %arg1, we know that %arg1 is the return value, and the
  // rest will be other loop carried variable
  IfOp ifOp = cast<IfOp>(breakOp->getParentOp());
  Value ifCond = ifOp.getOperand();
  Value start;
  Value end;
  // Position number in the BlockArgument list where the induction variable
  // is. Return empty value if we can't infer this number.
  std::optional<int64_t> inductionVarArgNumber;

  CmpOpMatcher matcher({mlir::index::IndexCmpPredicate::SLT,
                        mlir::index::IndexCmpPredicate::SGT});
  HLCF::ForLoopBoundCmpPredicate cmpPredicate;
  HLCF::ForLoopIndVarCompute indVarCompute;

  bool invertPred = (&ifOp.getThenRegion() == breakOp->getParentRegion());

  if (matcher.match(ifCond.getDefiningOp())) {
    mlir::index::CmpOp cmp = matcher.cmpOp;

    cmpPredicate = cmp.getPred() == mlir::index::IndexCmpPredicate::SLT
                       ? HLCF::ForLoopBoundCmpPredicate::SLT
                       : HLCF::ForLoopBoundCmpPredicate::SGT;

    // The operand who is a block argument is the induction variable, and its
    // initial value is the start value of the loop; the other operand (if a
    // constant) is the end of the loop.
    start =
        getValueIfConstAtLoopStart(cmp.getLhs(), inductionVarArgNumber, loop);
    if (inductionVarArgNumber.has_value()) {
      // The end must always be a constant in every iteration.
      end = getValueIfConstEveryIteration(cmp.getRhs(), loop, continueOp);
    } else {
      // No inductionVarArgNumber means `start` is not a block argument, and
      // comes directly from a const op. This means it is always constant every
      // iteration too. Can safely use it as `end`.
      end = start;
      start =
          getValueIfConstAtLoopStart(cmp.getRhs(), inductionVarArgNumber, loop);
      cmpPredicate = cmp.getPred() == mlir::index::IndexCmpPredicate::SLT
                         ? HLCF::ForLoopBoundCmpPredicate::SGT
                         : HLCF::ForLoopBoundCmpPredicate::SLT;
    }
  }

  if (!start || !end || !inductionVarArgNumber)
    return {};

  // Infer loop stride from ContinueOp's input operand expression.
  Value nextIter = continueOp.getOperand(inductionVarArgNumber.value());
  Value stride;
  Operation *nextIterOp = nextIter.getDefiningOp();
  if (isa<mlir::index::AddOp, mlir::index::SubOp>(nextIterOp)) {
    Value input0 = nextIterOp->getOperand(0);
    Value input1 = nextIterOp->getOperand(1);
    if (auto blockArg = dyn_cast<BlockArgument>(input0)) {
      stride = getValueIfConstEveryIteration(input1, loop, continueOp);
      indVarCompute = isa<mlir::index::AddOp>(nextIterOp)
                          ? HLCF::ForLoopIndVarCompute::ADD
                          : HLCF::ForLoopIndVarCompute::SUB;
    }
  }

  // Bail if we can't match pattern to find the stride value.
  if (!stride)
    return {};

  if (invertPred)
    cmpPredicate = invertCmpPred(cmpPredicate);

  return ForLoopBoundsAndSteps{start,        end,
                               stride,       inductionVarArgNumber.value(),
                               cmpPredicate, indVarCompute};
}

// Reorder values in the following order:
// 1. Value at inductionVarArgNumber.
// 2. Elements with indices in firstPartIndices.
// 3. Everything else in between.
static SmallVector<Value>
reorderValues(ValueRange values,
              const llvm::SetVector<int64_t> &firstPartIndices,
              int64_t inductionVarArgNumber) {
  SmallVector<Value> result;
  SmallVector<Value> secondPart;
  result.push_back(values[inductionVarArgNumber]);
  for (int64_t i : llvm::seq<int64_t>(0, values.size())) {
    if (!firstPartIndices.contains(i) && i != inductionVarArgNumber)
      secondPart.push_back(values[i]);
    else if (firstPartIndices.contains(i))
      result.push_back(values[i]);
  }

  llvm::append_range(result, secondPart);
  return result;
}

// Reorder values in the following order:
// 1. Value at inductionVarArgNumber.
// 2. Elements with indices in firstPartIndices.
// 3. Everything else in between.
// Each segment is a SmallVector so the hlcf.for.yield can use to create the
// operation.
static SmallVector<SmallVector<Value>>
reorderValueIntoGroups(ValueRange values,
                       const llvm::SetVector<int64_t> &firstPartIndices,
                       int64_t inductionVarArgNumber) {
  SmallVector<SmallVector<Value>> result(3);
  result[0].push_back(values[inductionVarArgNumber]);

  for (int64_t i = 0, e = values.size(); i != e; ++i) {
    if (!firstPartIndices.contains(i) && i != inductionVarArgNumber)
      result[2].push_back(values[i]);
    else if (firstPartIndices.contains(i))
      result[1].push_back(values[i]);
  }
  return result;
}

LogicalResult RaiseForLoops::raiseForLoops(LoopOp loop,
                                           InFlightDiagnostic &diag) {
  auto iter = loopJumpOps.find(loop);
  if (iter == loopJumpOps.end())
    return failure();

  // Only raise a loop with no early exits which should have only one BreakOp
  // and one ContinueOp.
  if (iter->second.size() <= 1) {
    diag.attachNote(loop->getLoc()) << "loop has no exit";
    return failure();
  }

  if (iter->second.size() > 2) {
    SmallVector<Operation *> breakOps;
    SmallVector<Operation *> continueOps;
    for (Operation *op : iter->second) {
      if (isa<BreakOp>(op))
        breakOps.push_back(op);
      else if (isa<ContinueOp>(op))
        continueOps.push_back(op);
    }

    if (breakOps.size() > 1 && continueOps.size() > 1) {
      diag.attachNote(loop->getLoc()) << "loop has multiple exits and multiple "
                                         "branches back to the beginning.";
    } else if (breakOps.size() > 1) {
      diag.attachNote(loop->getLoc()) << "loop has multiple exits";
    } else {
      diag.attachNote(loop->getLoc())
          << "loop has multiple branches back to the beginning.";
    }

    if (breakOps.size() > 1) {
      // Add diagnostics notes to each BreakOp in the loop.
      for (Operation *op : breakOps)
        diag.attachNote(op->getLoc()) << "loop exits";
    }

    if (continueOps.size() > 1) {
      // Add diagnostics notes to each ContinueOp in the loop.
      for (Operation *op : breakOps)
        diag.attachNote(op->getLoc()) << "loop branches back to the beginning";
    }
    return failure();
  }

  // Only raise a loop with no early exits which should have only one BreakOp
  // and one ContinueOp.
  BreakOp breakOp = dyn_cast<BreakOp>(iter->second.front());
  ContinueOp continueOp = dyn_cast<ContinueOp>(iter->second[!!breakOp]);
  if (!breakOp)
    breakOp = dyn_cast<BreakOp>(iter->second.back());
  if (!continueOp) {
    diag.attachNote(loop->getLoc()) << "cannot infer loop bounds and steps";
    return failure();
  }

  if (!breakOp) {
    diag.attachNote(loop->getLoc()) << "loop has no exit";
    return failure();
  }

  Block &body = loop.getBody().front();
  Operation *term = body.getTerminator();

  if (!isa<ContinueOp>(term)) {
    diag.attachNote(loop->getLoc()) << "cannot infer loop bounds and steps";
    return failure();
  }

  IfOp ifOp = dyn_cast<IfOp>(breakOp->getParentOp());

  if (!ifOp) {
    diag.attachNote(loop->getLoc()) << "cannot infer loop bounds and steps";
    return failure();
  }

  if (ifOp.getThenRegion().getBlocks().front().getOperations().size() != 1 ||
      ifOp.getElseRegion().getBlocks().front().getOperations().size() != 1) {
    // TODO: handle exit logic in loop unrolling and lower loops, which requires
    // raise ForOp to keep track of the exit block.
    diag.attachNote(loop->getLoc()) << "loop has complex exit logic";
    return failure();
  }

  std::optional<ForLoopBoundsAndSteps> loopInfo =
      inferLoopCount(loop, continueOp, breakOp);

  if (!loopInfo.has_value()) {
    diag.attachNote(loop->getLoc())
        << "cannot infer loop bounds and steps as constants for fully unroll";
    return failure();
  }

  mlir::IRRewriter rewriter{OpBuilder(loop)};

  // Collect return value arg numbers (indices).
  llvm::SetVector<int64_t> returnValueArgNumbers;
  for (auto op : breakOp.getOperands()) {
    if (auto arg = dyn_cast<BlockArgument>(op)) {
      returnValueArgNumbers.insert(arg.getArgNumber());
    } else {
      // Assuming that we only handle break has operands that are all
      // BlockArguments.
      diag.attachNote(loop->getLoc())
          << "complex loop structure, cannot infer loop bounds and steps";
      return failure();
    }
  }

  // Reorder loop operands to put return values first, and iterator last.
  SmallVector<Value> forOperands =
      reorderValues(loop->getOperands(), returnValueArgNumbers,
                    loopInfo->inductionVarArgNumber);

  // Create the new ForOp with reordered operands.
  auto forOp = rewriter.create<HLCF::ForOp>(
      loop->getLoc(), loop->getResultTypes(), loopInfo->lowerBound,
      loopInfo->upperBound, loopInfo->step, forOperands,
      loop.getUnrollLevelValue(), loopInfo->cmpPredicate,
      loopInfo->indVarCompute);

  // Create the block for the new ForOp.
  Block *block = rewriter.createBlock(&forOp.getBody());
  // Reorder block arguments and add them to the new block so that they match
  // ForOp's operands order.
  SmallVector<Value> reorderedArgs =
      reorderValues(body.getArguments(), returnValueArgNumbers,
                    loopInfo->inductionVarArgNumber);
  for (Value arg : reorderedArgs) {
    rewriter.replaceAllUsesWith(
        arg, block->addArgument(arg.getType(), arg.getLoc()));
  }

  Operation *prevOp = nullptr;
  for (Operation &op : llvm::make_early_inc_range(body.getOperations())) {
    if (&op == breakOp->getParentOp() && isa<IfOp>(op)) {
      // Don't move the parent IfOp of the break to the ForOp body.
      continue;
    }

    // Move op to the ForOp body.
    if (prevOp == nullptr) {
      // Move the first op to the beginning of the block.
      op.moveBefore(block, block->begin());
    } else {
      op.moveAfter(prevOp);
      if (auto c = dyn_cast<ContinueOp>(op)) {
        rewriter.setInsertionPointAfter(&op);

        // Reorder ContinueOp's operands to match ForOp's operand order
        // (return values first, loop interator last, and other loop carried
        // variables in between.)
        SmallVector<SmallVector<Value>> reorderedOperands =
            reorderValueIntoGroups(c.getOperands(), returnValueArgNumbers,
                                   loopInfo->inductionVarArgNumber);
        // Create `hlcf.for.yield` with the reordered operands.
        rewriter.create<HLCF::ForYieldOp>(
            op.getLoc(), reorderedOperands[0].front(), reorderedOperands[1],
            reorderedOperands[2]);
        c->dropAllReferences();
        rewriter.eraseOp(c);
      }
    }
    prevOp = &op;
  }

  loop->replaceAllUsesWith(forOp.getResults());

  // Erase the original loop.
  rewriter.eraseOp(loop);
  diag.abandon();

  return success();
}

void RaiseForLoops::runOnOperation() {
  loopJumpOps.clear();
  loopsToRaiseInOrder.clear();
  parentLoops.clear();

  walkLoopsPreorder(getOperation());
  // raise for-loops from inner to outer
  for (LoopOp loop : llvm::reverse(loopsToRaiseInOrder)) {
    // FIXME(#29784) https://github.com/modularml/modular/issues/29784
    // Revert this warning back to compilation error when we have more
    // sophisticated analysis to extra loop bounds and step info with general
    // patterns (e.g. SCEV). `@unroll` should be a compilation guarantee instead
    // of as a hint.
    InFlightDiagnostic diag =
        mlir::emitWarning(loop->getLoc(), " loop is decorated with @unroll, "
                                          "but compiler can't fully unroll it");

    if (failed(raiseForLoops(loop, diag)) && loop.isFullUnroll()) {
      if (!warnFailure) {
        // Don't warn failure because it's possible the loop can be raised if we
        // run this pass again.
        diag.abandon();
      }
      continue;
    }

    diag.abandon();
  }
}
