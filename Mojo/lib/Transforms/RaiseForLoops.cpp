//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Mojo/ToolCommon/KGENPasses.h"

#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "raise-for-loops"

using namespace M;
using namespace HLCF;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_RAISEFORLOOPS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

/// This pass has to run after elaboration so that all parameter
/// expressions have elaborated with known values.
// TODO: raise any LoopOp with or without an unroll factor if possible.
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
  LogicalResult raiseForLoops(LoopOp loop, InFlightDiagnostic &diag,
                              mlir::DominanceInfo &domInfo);
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

/// A value is loop-invariant if it dominates the loop. This is an O(1) check
/// and doesn't check if the value is transitively loop-invariant.
static bool isLoopInvariant(Value v, LoopOp loop,
                            mlir::DominanceInfo &domInfo) {
  Operation *parent = v.getDefiningOp();
  if (!parent)
    parent = cast<BlockArgument>(v).getParentBlock()->getParentOp();
  return domInfo.properlyDominates(parent, loop);
}

/// If `v` is a constant at the start of the loop, return that const value.
/// If `v` is passed in as the initial operand of the loop, argNumber is filled
/// with that operand number.
static Value getValueAtLoopEntry(Value v, std::optional<int64_t> &argNumber,
                                 LoopOp currLoop,
                                 mlir::DominanceInfo &domInfo) {
  // Check if the value is loop-invariant, and if so, return it.
  if (isLoopInvariant(v, currLoop, domInfo))
    return v;

  // Otherwise, check if this is an argument to the loop. If so, return its
  // value on entry.
  auto arg = dyn_cast<BlockArgument>(v);
  if (arg && arg.getParentBlock()->getParentOp() == currLoop) {
    // Save the argument number. This might be the loop induction variable.
    argNumber = arg.getArgNumber();
    return currLoop->getOperand(arg.getArgNumber());
  }

  // Otherwise, we don't know what this is.
  return {};
}

/// If `v` is a constant in every iteration of the loop, return that const
/// value. Requires `continueOp` to be the only continue in the loop.
static Value getValueIfLoopInvariant(Value v, LoopOp loop,
                                     ContinueOp continueOp,
                                     mlir::DominanceInfo &domInfo) {
  // Check if the value is loop-invariant, and if so, return it.
  if (isLoopInvariant(v, loop, domInfo))
    return v;

  // Check if this is an argument to the loop. If so, check if it remains the
  // same every iteration.
  auto arg = dyn_cast<BlockArgument>(v);
  if (arg && arg.getParentBlock()->getParentOp() == loop &&
      continueOp->getOperand(arg.getArgNumber()) == arg)
    return loop.getOperand(arg.getArgNumber());

  // Otherwise, we don't know what it is.
  return {};
}

// Match CmpOp with specific predicateTypes
class CmpOpMatcher {
public:
  CmpOpMatcher(const SmallVector<mlir::index::IndexCmpPredicate> &predTypes)
      : predicateTypes(predTypes) {}

  bool match(Operation *op) {
    if (auto c = dyn_cast_if_present<mlir::index::CmpOp>(op))
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

/// Matches the range-for idiom: an `hlcf.loop` with a `lit.try` whose `except`
/// breaks *this* loop (the iterator's StopIteration). Returns it, or null.
static LIT::TryOp matchRangeForIdiom(LoopOp loop) {
  StringAttr loopLabel = loop.getLabelAttr();
  for (Operation &op : loop.getBody().front()) {
    auto tryOp = dyn_cast<LIT::TryOp>(&op);
    if (!tryOp)
      continue;
    bool breaksThisLoop = false;
    tryOp.getExceptRegion().walk([&](BreakOp breakOp) {
      // Unlabelled break targets the nearest enclosing loop, i.e. this one.
      StringAttr breakLabel = breakOp.getLabelAttr();
      if (!breakLabel || breakLabel == loopLabel)
        breaksThisLoop = true;
    });
    if (breaksThisLoop)
      return tryOp;
  }
  return {};
}

/// Rewrites Mojo's exception-based `for i in range(n)` into the ascending
/// counted form raise-for-loops understands. `_ZeroStartingRange` counts up, so
/// the loop is `i=0`, StopIteration on `pop.cmp eq(i, n)` inside a `lit.try`
/// whose `except` breaks. This replaces that exit with `if (i < n) yield else
/// break` and feeds `continue` the `i + 1` increment. Ops preceding the raise
/// (loop-exit work, e.g. cleanup) ride the break path out of the loop. Bails
/// (no-op) on any shape mismatch or if a leading op uses a loop-internal value,
/// so raise-for-loops just skips the loop rather than miscompile.
static void normalizeRangeForIdiom(LoopOp loop, LIT::TryOp tryOp) {
  Block &body = loop.getBody().front();
  Location loc = loop.getLoc();

  // The StopIteration `raise` guarded by `pop.cmp eq(counter, n)`.
  LIT::TryRaiseOp raiseOp;
  tryOp.getTryRegion().walk([&](LIT::TryRaiseOp r) {
    if (!raiseOp)
      raiseOp = r;
  });
  if (!raiseOp)
    return;
  auto guardIf = raiseOp->getParentOfType<IfOp>();
  if (!guardIf || !guardIf.getElifRegions().empty())
    return;
  auto guardCmp = guardIf.getCond().getDefiningOp<POP::CmpOp>();
  if (!guardCmp || guardCmp.getPred() != KGEN::CmpPredicate::EQ)
    return;
  auto counterArg = dyn_cast<BlockArgument>(guardCmp.getLhs());
  if (!counterArg || counterArg.getOwner() != &body)
    return;
  unsigned cIdx = counterArg.getArgNumber();

  // Trip count `n` is the guard's other operand; must be loop invariant to be
  // the `i < n` bound. Counter must start at 0.
  Value n = guardCmp.getRhs();
  Operation *nDefiningOp = n.getDefiningOp();
  if (nDefiningOp ? loop->isAncestor(nDefiningOp) : n.getParentBlock() == &body)
    return;
  KGEN::SIMDAttr initAttr;
  if (!mlir::matchPattern(loop.getOperand(cIdx), mlir::m_Constant(&initAttr)) ||
      initAttr.getValues().size() != 1 ||
      !initAttr.getValues().front().getData().isZero())
    return;

  // The except break carries the loop's exit values.
  BreakOp breakOp;
  tryOp.getExceptRegion().walk([&](BreakOp b) {
    if (!breakOp)
      breakOp = b;
  });
  if (!breakOp)
    return;
  SmallVector<Value> breakOperands(breakOp.getOperands());
  // The break is rebuilt outside the `lit.try`, which is then erased, so an
  // operand defined inside it (the except block's exception payload) would be
  // left dangling. Bail rather than build a use of a value about to die.
  for (Value operand : breakOperands)
    if (tryOp->isAncestor(operand.getParentBlock()->getParentOp()))
      return;
  StringAttr label = loop.getLabelAttr();

  auto contOp = dyn_cast<ContinueOp>(body.getTerminator());
  if (!contOp)
    return;

  // `self.curr` bookkeeping (dead after the idiom is gone): the `curr + 1`
  // increment and the `curr if done else curr + 1` select. The increment also
  // confirms the count-up shape.
  POP::AddOp increment;
  POP::SelectOp curSelect;
  for (Operation &op : body) {
    if (auto add = dyn_cast<POP::AddOp>(&op)) {
      if (add.getLhs() == counterArg)
        increment = add;
    } else if (auto sel = dyn_cast<POP::SelectOp>(&op)) {
      if (sel.getCondition() == guardCmp.getResult())
        curSelect = sel;
    }
  }
  if (!increment)
    return;

  // The `lit.try`'s normal-path yield: positionally, `i` and the next counter.
  auto tryYield =
      dyn_cast<LIT::TryYieldOp>(tryOp.getTryRegion().front().getTerminator());
  if (!tryYield || tryYield.getNumOperands() != tryOp.getNumResults())
    return;

  // Post-normalization value for each try result: `i` -> counter, next counter
  // -> `i + 1`. Bail on any other carried value.
  SmallVector<Value> resultReplacements;
  resultReplacements.reserve(tryOp.getNumResults());
  for (Value yielded : tryYield.getOperands()) {
    if (yielded == counterArg)
      resultReplacements.push_back(counterArg);
    else if (yielded == increment.getResult() ||
             (curSelect && yielded == curSelect.getResult()))
      resultReplacements.push_back(increment.getResult());
    else
      return;
  }

  // Ops before the raise run on the exit path and get hoisted with the break;
  // reject any using a loop-internal value (can't move past the loop) with a
  // soft error. Track results so a leading op may feed a later one.
  Block *raiseBlock = raiseOp->getBlock();
  DenseSet<Value> movableResults;
  for (Operation &leadingOp : raiseBlock->without_terminator()) {
    for (Value operand : leadingOp.getOperands()) {
      if (movableResults.contains(operand))
        continue;
      Operation *defScope = operand.getParentBlock()->getParentOp();
      if (loop->isAncestor(defScope)) {
        InFlightDiagnostic diag = mlir::emitWarning(leadingOp.getLoc())
                                  << "raise-for-loops: loop-exit op depends on "
                                     "a loop-internal value; cannot hoist it "
                                     "out of the loop";
        bool dropDiag = true;
        LLVM_DEBUG(dropDiag = false;);
        if (dropDiag)
          diag.abandon();
        return;
      }
    }
    for (Value result : leadingOp.getResults())
      movableResults.insert(result);
  }

  OpBuilder b(loop.getContext());

  // `if (i < n) { yield } else { <exit ops>; break }` at the top of the body.
  b.setInsertionPointToStart(&body);
  Value cond =
      POP::CmpOp::create(b, loc, KGEN::CmpPredicate::LT, counterArg, n);
  auto ifOp = IfOp::create(b, loc, TypeRange{}, cond);
  b.createBlock(&ifOp.getThenRegion());
  YieldOp::create(b, loc);
  Block *elseBlock = b.createBlock(&ifOp.getElseRegion());
  // The exit-path ops precede the break; the raise below relocates the whole
  // break block after the loop.
  for (Operation &leadingOp :
       llvm::make_early_inc_range(raiseBlock->without_terminator()))
    leadingOp.moveBefore(elseBlock, elseBlock->end());
  b.setInsertionPointToEnd(elseBlock);
  BreakOp::create(b, loc, breakOperands, label);

  // Rewire try results, then drop the `lit.try` and now-dead bookkeeping.
  for (auto [res, repl] : llvm::zip(tryOp.getResults(), resultReplacements))
    res.replaceAllUsesWith(repl);
  tryOp.erase();
  auto eraseIfDead = [](Operation *op) {
    if (op && op->use_empty())
      op->erase();
  };
  eraseIfDead(curSelect);
  eraseIfDead(guardCmp);
}

static std::optional<ForLoopBoundsAndSteps>
inferLoopCount(LoopOp loop, ContinueOp continueOp, BreakOp breakOp,
               mlir::DominanceInfo &domInfo) {
  // The infer logic here is assuming that for-loop's ranges are:
  // - range(n) - zero starting range with start = 0, end = n, stride = 1.
  // - range(s, e) - sequential range with start = s, end = e, stride = 1.
  // - range(s, e, st) - strided range with start = s, end = e, stride = st.
  // This is pretty limited assumption to bootstrap loop unrolling.
  // This can be improved to support more general for loops.

  // Infer loop start and end from BreakOp's parent if/elif operand expression.
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
  Operation *parent = breakOp->getParentOp();
  Value ifCond;
  Region *thenRegion = nullptr;
  auto ifOp = dyn_cast<IfOp>(parent);
  // Multi-arm elif is expanded before this pass; only 2-region if/else shape
  // is recognized as a loop exit test.
  if (!ifOp || ifOp.getNumRegions() != 2)
    return {};
  ifCond = ifOp.getCond();
  thenRegion = &ifOp.getThenRegion();

  // `pop.cast_from_builtin`. Look through that cast to recognize  `index.cmp`
  // TODO: we won't need this after migrating scalar<int>, but the pattern
  // matching below needs to be updated as well.
  if (auto castOp =
          dyn_cast_if_present<POP::CastFromBuiltinOp>(ifCond.getDefiningOp()))
    ifCond = castOp.getInput();
  Value start;
  Value end;
  // Position number in the BlockArgument list where the induction variable
  // is. Return empty value if we can't infer this number.
  std::optional<int64_t> inductionVarArgNumber;

  CmpOpMatcher matcher({mlir::index::IndexCmpPredicate::SLT,
                        mlir::index::IndexCmpPredicate::SGT});
  HLCF::ForLoopBoundCmpPredicate cmpPredicate;
  HLCF::ForLoopIndVarCompute indVarCompute;

  bool invertPred = (thenRegion == breakOp->getParentRegion());

  if (matcher.match(ifCond.getDefiningOp())) {
    mlir::index::CmpOp cmp = matcher.cmpOp;

    cmpPredicate = cmp.getPred() == mlir::index::IndexCmpPredicate::SLT
                       ? HLCF::ForLoopBoundCmpPredicate::SLT
                       : HLCF::ForLoopBoundCmpPredicate::SGT;

    // The operand who is a block argument is the induction variable, and its
    // initial value is the start value of the loop; the other operand (if a
    // constant) is the end of the loop.
    start =
        getValueAtLoopEntry(cmp.getLhs(), inductionVarArgNumber, loop, domInfo);
    if (inductionVarArgNumber.has_value()) {
      // The end must always be a constant in every iteration.
      end = getValueIfLoopInvariant(cmp.getRhs(), loop, continueOp, domInfo);
    } else {
      // No inductionVarArgNumber means `start` is not a block argument, and
      // comes directly from a const op. This means it is always constant every
      // iteration too. Can safely use it as `end`.
      end = start;
      start = getValueAtLoopEntry(cmp.getRhs(), inductionVarArgNumber, loop,
                                  domInfo);
      cmpPredicate = cmp.getPred() == mlir::index::IndexCmpPredicate::SLT
                         ? HLCF::ForLoopBoundCmpPredicate::SGT
                         : HLCF::ForLoopBoundCmpPredicate::SLT;
    }
  } else if (auto cmp =
                 dyn_cast_if_present<POP::CmpOp>(ifCond.getDefiningOp())) {
    KGEN::CmpPredicate pred = cmp.getPred();
    if (pred != KGEN::CmpPredicate::LT && pred != KGEN::CmpPredicate::GT)
      return {};
    auto simdTy = dyn_cast<SIMDType>(cmp.getLhs().getType());
    auto dtype = simdTy ? simdTy.getResolvedDType() : std::nullopt;

    // For now allow signed integers only, which is aligned to code above that
    // pattern-matches `index`-ops.
    if (!dtype || !dtype->isSInt())
      return {};
    cmpPredicate = pred == KGEN::CmpPredicate::LT
                       ? HLCF::ForLoopBoundCmpPredicate::SLT
                       : HLCF::ForLoopBoundCmpPredicate::SGT;

    start =
        getValueAtLoopEntry(cmp.getLhs(), inductionVarArgNumber, loop, domInfo);
    if (inductionVarArgNumber.has_value()) {
      end = getValueIfLoopInvariant(cmp.getRhs(), loop, continueOp, domInfo);
    } else {
      end = start;
      start = getValueAtLoopEntry(cmp.getRhs(), inductionVarArgNumber, loop,
                                  domInfo);
      cmpPredicate = pred == KGEN::CmpPredicate::LT
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
  if (isa<mlir::index::AddOp, mlir::index::SubOp, POP::AddOp, POP::SubOp>(
          nextIterOp)) {
    Value input0 = nextIterOp->getOperand(0);
    Value input1 = nextIterOp->getOperand(1);
    if (auto blockArg = dyn_cast<BlockArgument>(input0)) {
      stride = getValueIfLoopInvariant(input1, loop, continueOp, domInfo);
      indVarCompute = isa<mlir::index::AddOp, POP::AddOp>(nextIterOp)
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

/// Return whether the if/elif exit test has complex logic that is not supported
/// for raising.
template <typename IfLikeOp>
static bool hasComplexExitLogic(LoopOp loop, IfLikeOp ifOp) {
  // The if op is known to have one yield and one break.
  Block *breakBlock = &ifOp.getThenRegion().getBlocks().front();
  Block *yieldBlock = &ifOp.getElseRegion().getBlocks().front();
  if (!isa<BreakOp>(breakBlock->getTerminator()))
    std::swap(breakBlock, yieldBlock);

  // Currently do not support complex yield block.
  if (!yieldBlock->without_terminator().empty())
    return true;

  // Loops that have try-except operations (including nested loops) cannot be
  // transformed into SESE loops as it's an early exit.
  // NOTE: that code assumes `lit.try` operation without a `lit.try.raise` was
  // simplified before.
  if (ifOp->walk([&](Operation *op) {
            if (isa<LIT::TryOp>(op))
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return true;

  // Break block can contain ops that do not depend on values internal to the
  // parent loop. These ops can be moved to after the for-loop because this
  // break is the only exit for the loop.
  DenseSet<Value> intermediateValues;
  for (Operation &op : breakBlock->without_terminator()) {
    for (Value operand : op.getOperands()) {
      if (intermediateValues.contains(operand))
        continue;

      // If value is a block arg, this gets the defining block's parent
      // op. Otherwise this gets the parent of the defining op.
      Operation *closestDefiningScope = operand.getParentBlock()->getParentOp();
      // If the parent loop is closestDefiningScope, or if it contains
      // closestDefiningScope, this dbgValue describes a loop-internal value.
      if (loop->isAncestor(closestDefiningScope))
        return true;
    }
    intermediateValues.insert(op.result_begin(), op.result_end());
  }

  // BreakOp must not rely on any of the intermediate values.
  return llvm::any_of(
      breakBlock->getTerminator()->getOperands(),
      [&](Value operand) { return intermediateValues.contains(operand); });
}

LogicalResult RaiseForLoops::raiseForLoops(LoopOp loop,
                                           InFlightDiagnostic &diag,
                                           mlir::DominanceInfo &domInfo) {
  auto iter = loopJumpOps.find(loop);
  if (iter == loopJumpOps.end())
    return failure();

  // Only raise a loop with no early exits which should have only one BreakOp
  // and one ContinueOp.
  if (iter->second.size() <= 1)
    return diag.attachNote(loop->getLoc()) << "loop has no exit";

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
    return diag.attachNote(loop->getLoc())
           << "cannot infer loop bounds and steps";
  }

  if (!breakOp)
    return diag.attachNote(loop->getLoc()) << "loop has no exit";

  Block &body = loop.getBody().front();
  Operation *term = body.getTerminator();

  if (!isa<ContinueOp>(term)) {
    return diag.attachNote(loop->getLoc())
           << "cannot infer loop bounds and steps";
  }

  Operation *ifLikeParent = breakOp->getParentOp();
  auto ifOp = dyn_cast<IfOp>(ifLikeParent);
  if (!ifOp || ifOp.getNumRegions() != 2) {
    return diag.attachNote(loop->getLoc())
           << "cannot infer loop bounds and steps";
  }

  if (hasComplexExitLogic(loop, ifOp)) {
    // TODO: handle exit logic in loop unrolling and lower loops, which requires
    // raise ForOp to keep track of the exit block.
    return diag.attachNote(loop->getLoc()) << "loop has complex exit logic";
  }

  std::optional<ForLoopBoundsAndSteps> loopInfo =
      inferLoopCount(loop, continueOp, breakOp, domInfo);

  if (!loopInfo.has_value()) {
    return diag.attachNote(loop->getLoc())
           << "cannot infer loop bounds and steps as constants for fully "
              "unroll";
  }

  IRRewriter rewriter{OpBuilder(loop)};

  mlir::IndexType indexTy = rewriter.getIndexType();
  auto *ctx = loop->getContext();
  SIMDType popIndexTy = SIMDType::get(ctx, 1, KGENDType(KGENDType::index));

  // `ForControlVariableType` accepts `index` or `!kgen.scalar<signed integer>`,
  // so a scalar loop needs no cast at all. Forcing one round-trips the
  // induction variable through `pop.cast` on entry and again at the yield,
  // every iteration, which hides its value range from later analyses: address
  // arithmetic loses its disjoint-`or` form and the copies stay live, costing
  // registers a 1024-thread launch does not have.
  auto isControlVariableType = [&](Type t) {
    if (t == indexTy)
      return true;
    auto simd = dyn_cast<SIMDType>(t);
    return simd && simd.isScalar() && simd.getResolvedDType().has_value() &&
           simd.getResolvedDType()->isSInt();
  };

  // The verifier requires the three bounds to share one type, and the body's
  // induction argument and its initial value ride along with them. Anything
  // less than full agreement falls back to `index`, where the casts below
  // reconcile the types.
  unsigned indVarNo = loopInfo->inductionVarArgNumber;
  Type controlTy = indexTy;
  Type boundsTy = loopInfo->lowerBound.getType();
  if (boundsTy == loopInfo->upperBound.getType() &&
      boundsTy == loopInfo->step.getType() &&
      boundsTy == loop->getOperand(indVarNo).getType() &&
      boundsTy == body.getArgument(indVarNo).getType() &&
      isControlVariableType(boundsTy))
    controlTy = boundsTy;

  auto castToControlTy = [&](Value v) -> Value {
    if (v.getType() == controlTy)
      return v;
    rewriter.setInsertionPoint(loop);
    Value cast = POP::CastOp::create(rewriter, loop->getLoc(), popIndexTy, v);
    return POP::CastToBuiltinOp::create(rewriter, loop->getLoc(), indexTy,
                                        cast);
  };
  loopInfo->lowerBound = castToControlTy(loopInfo->lowerBound);
  loopInfo->upperBound = castToControlTy(loopInfo->upperBound);
  loopInfo->step = castToControlTy(loopInfo->step);

  // Collect return value arg numbers (indices).
  llvm::SetVector<int64_t> returnValueArgNumbers;
  for (auto op : breakOp.getOperands()) {
    if (auto arg = dyn_cast<BlockArgument>(op)) {
      returnValueArgNumbers.insert(arg.getArgNumber());
    } else {
      // Assuming that we only handle break has operands that are all
      // BlockArguments.
      return diag.attachNote(loop->getLoc())
             << "complex loop structure, cannot infer loop bounds and steps";
    }
  }

  // Reorder loop operands to put return values first, and iterator last.
  SmallVector<Value> forOperands =
      reorderValues(loop->getOperands(), returnValueArgNumbers,
                    loopInfo->inductionVarArgNumber);

  // The first forOperand is the initial value for the induction variable block
  // argument, which must match the type the bounds settled on.
  if (!forOperands.empty() && forOperands[0].getType() != controlTy) {
    rewriter.setInsertionPoint(loop);
    Value cast = POP::CastOp::create(rewriter, loop->getLoc(), popIndexTy,
                                     forOperands[0]);
    forOperands[0] =
        POP::CastToBuiltinOp::create(rewriter, loop->getLoc(), indexTy, cast);
  }

  // Create the new ForOp with reordered operands.
  auto forOp = HLCF::ForOp::create(
      rewriter, loop->getLoc(), loop->getResultTypes(), loopInfo->lowerBound,
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

  // Only when the body's induction argument disagrees with the loop's control
  // type does it need converting back at the top of the block.
  Type origIndVarType = reorderedArgs[0].getType();
  Operation *lastInsertedOp = nullptr;
  for (auto [i, arg] : llvm::enumerate(reorderedArgs)) {
    if (i == 0 && origIndVarType != controlTy) {
      Value idxArg = block->addArgument(controlTy, arg.getLoc());
      rewriter.setInsertionPointToStart(block);
      Value fromBuiltin = POP::CastFromBuiltinOp::create(rewriter, arg.getLoc(),
                                                         popIndexTy, idxArg);
      Value asOrigType = POP::CastOp::create(rewriter, arg.getLoc(),
                                             origIndVarType, fromBuiltin);
      lastInsertedOp = asOrigType.getDefiningOp();
      rewriter.replaceAllUsesWith(arg, asOrigType);
    } else {
      rewriter.replaceAllUsesWith(
          arg, block->addArgument(arg.getType(), arg.getLoc()));
    }
  }

  Operation *prevOp = nullptr;
  for (Operation &op : llvm::make_early_inc_range(body.getOperations())) {
    if (&op == breakOp->getParentOp() && isa<IfOp>(op)) {
      // Don't move the parent elif of the break to the ForOp body.
      continue;
    }

    // Move op to the ForOp body.
    if (prevOp == nullptr) {
      // Move the first op after any pre-inserted cast ops (or to block begin
      // if there are none).
      if (lastInsertedOp)
        op.moveAfter(lastInsertedOp);
      else
        op.moveBefore(block, block->begin());
    } else {
      op.moveAfter(prevOp);
      if (auto c = dyn_cast<ContinueOp>(op)) {
        rewriter.setInsertionPointAfter(&op);

        // Reorder ContinueOp's operands to match ForOp's operand order
        // (return values first, loop iterator last, and other loop carried
        // variables in between.)
        SmallVector<SmallVector<Value>> reorderedOperands =
            reorderValueIntoGroups(c.getOperands(), returnValueArgNumbers,
                                   loopInfo->inductionVarArgNumber);

        // Cast the next induction value back only if the body works in a
        // different type than the loop's control type.
        if (origIndVarType != controlTy) {
          Value nextIndVar = reorderedOperands[0].front();
          Value castToIdx = POP::CastOp::create(rewriter, op.getLoc(),
                                                popIndexTy, nextIndVar);
          reorderedOperands[0][0] = POP::CastToBuiltinOp::create(
              rewriter, op.getLoc(), indexTy, castToIdx);
        }

        // Create `hlcf.for.yield` with the reordered operands.
        HLCF::ForYieldOp::create(rewriter, op.getLoc(),
                                 reorderedOperands[0].front(),
                                 reorderedOperands[1], reorderedOperands[2]);
        c->dropAllReferences();
        rewriter.eraseOp(c);
      }
    }
    prevOp = &op;
  }

  // Move operations in the break-branch of `ifOp` immediately after the loop.
  rewriter.inlineBlockBefore(breakOp->getBlock(), forOp->getNextNode());
  breakOp->erase();

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

  // Normalize `for i in range(n)` to the counted form before analysis (collect
  // first, since the rewrite mutates the IR).
  {
    SmallVector<std::pair<LoopOp, LIT::TryOp>> rangeForLoops;
    getOperation()->walk([&](LoopOp loop) {
      if (LIT::TryOp tryOp = matchRangeForIdiom(loop))
        rangeForLoops.push_back({loop, tryOp});
    });
    for (auto [loop, tryOp] : rangeForLoops)
      normalizeRangeForIdiom(loop, tryOp);
  }

  auto &domInfo = getAnalysis<mlir::DominanceInfo>();

  walkLoopsPreorder(getOperation());
  // raise for-loops from inner to outer
  for (LoopOp loop : llvm::reverse(loopsToRaiseInOrder)) {
    InFlightDiagnostic diag =
        mlir::emitWarning(loop->getLoc(), "failed to raise loop");
    (void)raiseForLoops(loop, diag, domInfo);
    bool dropDiag = true;
    LLVM_DEBUG(dropDiag = false;);
    if (dropDiag)
      diag.abandon();
  }
}
