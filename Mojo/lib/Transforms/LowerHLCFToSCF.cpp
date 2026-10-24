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
//
// Lower HLCF structured control flow to SCF. Run `raise-for-loops` first for
// `for i in range(...)` to reach `scf.for` rather than `scf.while`.
//
//===----------------------------------------------------------------------===//

#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/Compiler/DirectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERHLCFTOSCF
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

//===----------------------------------------------------------------------===//
// Shared guards
//===----------------------------------------------------------------------===//

/// Whether `op` holds an `hlcf.return` in its regions. It stays legal for the
/// function conversion, so a region ending in one would leave it terminating
/// the new `scf` op, which wants an `scf.yield`. `hlcf.unreachable` is not an
/// exit; `replaceTrailingUnreachable` handles it.
static bool hasNestedFunctionExit(Operation *op) {
  return op
      ->walk<mlir::WalkOrder::PreOrder>([op](Operation *inner) {
        // An inlined async region or staged closure is its own function; the
        // `hlcf.return` terminating it leaves that one, not `op`.
        if (inner != op && isa<KGEN::FunctionLike>(inner))
          return mlir::WalkResult::skip();
        return isa<HLCF::ReturnOp>(inner) ? mlir::WalkResult::interrupt()
                                          : mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

/// Append `values` to `out`, bridging each to its slot in `types`. The loop
/// terminators are built from body values the conversion has not rewritten
/// yet, so without this the `scf.while` state types would not line up.
/// Replace a region-ending `hlcf.unreachable` with the yield the enclosing
/// `scf` op needs. Control never gets there, so the values are poison.
static void replaceTrailingUnreachable(DirectRewriter &rewriter, Location loc,
                                       Block *block, TypeRange resultTypes) {
  auto unreachable = dyn_cast<HLCF::UnreachableOp>(block->getTerminator());
  if (!unreachable)
    return;
  rewriter.setInsertionPoint(unreachable);
  SmallVector<Value> poison;
  for (Type t : resultTypes)
    poison.push_back(ub::PoisonOp::create(rewriter, loc, t));
  rewriter.replaceOpWithNewOp<scf::YieldOp>(unreachable, poison);
}

/// The constant value of `step`, or nullopt when it is not a constant.
static std::optional<int64_t> constantStep(Value step) {
  mlir::IntegerAttr attr;
  if (mlir::matchPattern(step, mlir::m_Constant(&attr)))
    return attr.getInt();
  return std::nullopt;
}

/// The first `continue`/`break` of `loop` sitting below the loop body, or
/// null. Exits inside a nested loop belong to that loop.
static Operation *findExitBelowBody(HLCF::LoopOp loop) {
  Operation *found = nullptr;
  loop.getBody().walk<mlir::WalkOrder::PreOrder>([&](Operation *inner) {
    if (isa<HLCF::LoopOp, HLCF::ForOp>(inner))
      return mlir::WalkResult::skip();
    if (isa<HLCF::ContinueOp, HLCF::BreakOp>(inner) &&
        inner->getBlock()->getParent() != &loop.getBody()) {
      found = inner;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

/// Whether `yield` advances the induction variable the way `scf.for` will:
/// `iv +/- step` on the loop's own block argument, matching the declared
/// compute kind.
static bool yieldsStepAdvancedInductionVar(HLCF::ForOp forOp,
                                           HLCF::ForYieldOp yield) {
  Value next = yield.getInductionVar();
  Value iv = forOp.getBody().front().getArgument(0);
  // A loop that runs its induction variable straight through is fine: the
  // `scf.for` body argument already holds the right value.
  if (next == iv)
    return true;

  Operation *def = next.getDefiningOp();
  if (!def)
    return false;
  bool isAdd = isa<mlir::index::AddOp, mlir::arith::AddIOp>(def);
  bool isSub = isa<mlir::index::SubOp, mlir::arith::SubIOp>(def);
  bool wantSub =
      forOp.getIndVarComputeType() == HLCF::ForLoopIndVarCompute::SUB;
  if ((wantSub && !isSub) || (!wantSub && !isAdd))
    return false;
  return def->getNumOperands() == 2 && def->getOperand(0) == iv &&
         def->getOperand(1) == forOp.getStep();
}

static void appendConverted(DirectRewriter &rewriter, Location loc,
                            ValueRange values, TypeRange types,
                            SmallVectorImpl<Value> &out) {
  for (auto [value, type] : llvm::zip(values, types)) {
    if (value.getType() == type) {
      out.push_back(value);
      continue;
    }
    out.push_back(
        mlir::UnrealizedConversionCastOp::create(rewriter, loc, type, value)
            .getResult(0));
  }
}

/// Bridge `values` back to the types `block`'s arguments have, so merging them
/// in leaves the still-HLCF ops well typed. The direct conversion driver
/// inherits `mergeBlocks` from `RewriterBase`, which substitutes the values
/// as given; `ConversionPatternRewriter` materializes these casts itself.
static SmallVector<Value> bridgedForMerge(DirectRewriter &rewriter,
                                          Location loc, Block &block,
                                          Block *dest, ValueRange values) {
  // The casts take `dest`'s own block arguments, so they have to sit inside
  // `dest` to dominate their uses -- and after whatever the caller already
  // built there, which they may consume.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(dest);
  SmallVector<Value> out;
  for (auto [i, value] : llvm::enumerate(values)) {
    Type want = block.getArgument(i).getType();
    out.push_back(value.getType() == want
                      ? value
                      : mlir::UnrealizedConversionCastOp::create(rewriter, loc,
                                                                 want, value)
                            .getResult(0));
  }
  return out;
}

/// Bridge `cond` to `i1` for an `scf.if`. An elif condition is read out of a
/// region the conversion has not rewritten yet, so it still carries its
/// `!kgen.scalar<bool>` type.
static Value bridgeToI1(DirectRewriter &rewriter, Location loc, Value cond) {
  if (cond.getType().isInteger(1))
    return cond;
  return mlir::UnrealizedConversionCastOp::create(rewriter, loc,
                                                  rewriter.getI1Type(), cond)
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// hlcf.for → scf.for
//
// HLCF body block args: [iv, rv..., lc...]  (iterArgs[0]=IV, rest=loop-carried)
// SCF  body block args: [iv, rv..., lc...]  (same layout)
// SCF initArgs = hlcf.iterArgs[1:] (IV is managed by scf.for via lowerBound)
//===----------------------------------------------------------------------===//

struct ConvertHLCFFor : public DirectOpConversionPattern<HLCF::ForOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  /// Returns `v`, or an equivalent `arith.constant` when `v` is an
  /// `index.constant`. Mojo emits `range(N)`'s upper bound as
  /// `index.constant N`, and some backends (the NISA → BIR translator) only
  /// consume loop bounds that are `arith` constants.
  static Value asArithConstant(DirectRewriter &rewriter, Value v) {
    if (auto idxConst = v.getDefiningOp<mlir::index::ConstantOp>()) {
      return arith::ConstantOp::create(rewriter, idxConst.getLoc(),
                                       idxConst.getValueAttr());
    }
    return v;
  }

  LogicalResult matchAndRewrite(HLCF::ForOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (hasNestedFunctionExit(op))
      return rewriter.notifyMatchFailure(
          op, "a nested `return` has no structured-control-flow form");
    // `scf.for` drives the induction variable from `step`, so a body that
    // advances it any other way would silently change the trip count.
    auto yieldOp =
        dyn_cast<HLCF::ForYieldOp>(op.getBody().front().getTerminator());
    if (!yieldOp || !yieldsStepAdvancedInductionVar(op, yieldOp))
      return rewriter.notifyMatchFailure(
          op, "the induction variable is not advanced by the loop's `step`");

    // `scf.for` is ascending, so a descending loop runs the same span and
    // recovers its induction value as `iv = lb + ub - k`. A negative step
    // with `sgt add` descends too, once negated.
    auto pred = op.getCmpPredicateType();
    bool downcounting =
        op.getIndVarComputeType() == HLCF::ForLoopIndVarCompute::SUB;
    Value negatedStep;
    if (!downcounting && pred == HLCF::ForLoopBoundCmpPredicate::SGT) {
      if (std::optional<int64_t> stepConst = constantStep(op.getStep()))
        if (*stepConst < 0) {
          downcounting = true;
          negatedStep = arith::ConstantOp::create(
              rewriter, op.getLoc(),
              rewriter.getIntegerAttr(rewriter.getIndexType(), -*stepConst));
        }
    }

    // An inclusive bound (`sle`/`sge`) would drop its last iteration, so
    // reject every other pair rather than silently miscompiling it.
    bool halfOpen = downcounting ? pred == HLCF::ForLoopBoundCmpPredicate::SGT
                                 : pred == HLCF::ForLoopBoundCmpPredicate::SLT;
    if (!halfOpen)
      return rewriter.notifyMatchFailure(
          op, "only half-open for-loops (ascending `slt`, descending `sgt`) "
              "are supported");

    Value origLb = adaptor.getLowerBound();
    Value origUb = adaptor.getUpperBound();
    Value lb = asArithConstant(rewriter, downcounting ? origUb : origLb);
    Value ub = asArithConstant(rewriter, downcounting ? origLb : origUb);
    Value step = negatedStep ? negatedStep
                             : asArithConstant(rewriter, adaptor.getStep());

    ValueRange iterArgs = adaptor.getIterArgs();
    auto scfFor = scf::ForOp::create(rewriter, op.getLoc(), lb, ub, step,
                                     iterArgs.drop_front(),
                                     /*bodyBuilder=*/nullptr);

    // scf.for auto-inserts a terminator scf.yield when no body builder is
    // given. Erase it so merging the HLCF body (which brings hlcf.for.yield,
    // converted later) doesn't leave two terminators.
    Block *scfBody = scfFor.getBody();
    if (!scfBody->empty())
      rewriter.eraseOp(&scfBody->back());

    SmallVector<Value> mergedArgs(scfBody->getArguments().begin(),
                                  scfBody->getArguments().end());
    if (downcounting) {
      rewriter.setInsertionPointToStart(scfBody);
      // `arith`, not `index`: `ForControlVariableType` also admits
      // `!kgen.scalar<siN>` bounds, which convert to `iN`.
      Value sum = arith::AddIOp::create(rewriter, op.getLoc(), origLb, origUb);
      mergedArgs[0] = arith::SubIOp::create(rewriter, op.getLoc(), sum,
                                            scfBody->getArgument(0));
    }
    rewriter.mergeBlocks(&op.getBody().front(), scfBody,
                         bridgedForMerge(rewriter, op.getLoc(),
                                         op.getBody().front(), scfBody,
                                         mergedArgs));

    // `hlcf.for` results are the retvals only, while `scf.for` also returns
    // the other loop-carried values.
    rewriter.replaceOp(op, scfFor.getResults().take_front(op.getNumResults()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// hlcf.for.yield → scf.yield (retvals ++ otherIterValues, no induction var)
//===----------------------------------------------------------------------===//

struct ConvertHLCFForYield
    : public DirectOpConversionPattern<HLCF::ForYieldOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(HLCF::ForYieldOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    SmallVector<Value> yieldVals;
    llvm::append_range(yieldVals, adaptor.getReturnValues());
    llvm::append_range(yieldVals, adaptor.getOtherIterValues());
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, yieldVals);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// hlcf.if → scf.if
//===----------------------------------------------------------------------===//

struct ConvertHLCFIf : public DirectOpConversionPattern<HLCF::IfOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(HLCF::IfOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (hasNestedFunctionExit(op))
      return rewriter.notifyMatchFailure(
          op, "a nested `return` has no structured-control-flow form");

    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return mlir::failure();

    // `elifRegions` holds (condition, then) pairs. `scf.if` is two-armed, so
    // each pair becomes another `scf.if` in the previous one's else arm, with
    // the condition region's own ops inlined ahead of it to compute its
    // condition. The last else arm takes the original `hlcf.if` else region.
    auto elifs = op.getElifRegions();
    Location loc = op.getLoc();

    auto scfIf = scf::IfOp::create(rewriter, loc, resultTypes,
                                   adaptor.getCond(), /*withElseRegion=*/true);
    // The builder inserts an implicit `scf.yield` in each block; drop it
    // before merging so the block does not end up with two terminators.
    auto clearTerminator = [&](Block *b) {
      if (!b->empty())
        rewriter.eraseOp(&b->back());
    };
    clearTerminator(scfIf.thenBlock());
    rewriter.mergeBlocks(&op.getThenRegion().front(), scfIf.thenBlock(), {});
    replaceTrailingUnreachable(rewriter, loc, scfIf.thenBlock(), resultTypes);

    scf::IfOp innermost = scfIf;
    // A cond-yield passes its extra operands to both of its targets: the
    // paired `then` and whatever the false edge runs next. So the values a
    // region takes as block arguments come from the arm *before* it.
    SmallVector<Value> incoming;
    for (unsigned i = 0, e = elifs.size(); i < e; i += 2) {
      Block &condBlock = elifs[i].front();
      Block &thenBlock = elifs[i + 1].front();
      auto condYield = cast<HLCF::IfElifCondYieldOp>(condBlock.getTerminator());
      Value cond = condYield.getOperands().front();
      SmallVector<Value> carried(condYield.getOperands().begin() + 1,
                                 condYield.getOperands().end());

      Block *elseBlock = innermost.elseBlock();
      clearTerminator(elseBlock);
      // Inline the condition region's own ops, so the condition is computed
      // only once the earlier arms have been ruled out.
      rewriter.mergeBlocks(&condBlock, elseBlock, incoming);
      rewriter.setInsertionPointToEnd(elseBlock);
      Value bridgedCond = bridgeToI1(rewriter, loc, cond);
      rewriter.eraseOp(condYield);

      auto nested = scf::IfOp::create(rewriter, loc, resultTypes, bridgedCond,
                                      /*withElseRegion=*/true);
      // The nested `scf.if` is the whole value of the arm holding it.
      scf::YieldOp::create(rewriter, loc, nested.getResults());

      clearTerminator(nested.thenBlock());
      rewriter.mergeBlocks(&thenBlock, nested.thenBlock(), carried);
      innermost = nested;
      incoming = std::move(carried);
    }

    Block *lastElse = innermost.elseBlock();
    clearTerminator(lastElse);
    rewriter.mergeBlocks(&op.getElseRegion().front(), lastElse, incoming);
    replaceTrailingUnreachable(rewriter, loc, lastElse, resultTypes);

    rewriter.replaceOp(op, scfIf.getResults());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// hlcf.yield → scf.yield
//===----------------------------------------------------------------------===//

struct ConvertHLCFYield : public DirectOpConversionPattern<HLCF::YieldOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(HLCF::YieldOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// hlcf.switch → scf.index_switch
//===----------------------------------------------------------------------===//

struct ConvertHLCFSwitch : public DirectOpConversionPattern<HLCF::SwitchOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(HLCF::SwitchOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (hasNestedFunctionExit(op))
      return rewriter.notifyMatchFailure(
          op, "a nested `return` has no structured-control-flow form");
    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return mlir::failure();

    SmallVector<int64_t> caseValues(op.getCaseValues().begin(),
                                    op.getCaseValues().end());
    auto scfSwitch = scf::IndexSwitchOp::create(
        rewriter, op.getLoc(), resultTypes, adaptor.getArg(), caseValues,
        /*numCaseRegions=*/op.getCaseRegions().size());

    rewriter.inlineRegionBefore(op.getDefaultRegion(),
                                scfSwitch.getDefaultRegion(),
                                scfSwitch.getDefaultRegion().end());
    if (failed(rewriter.convertRegionTypes(&scfSwitch.getDefaultRegion(),
                                           *getTypeConverter())))
      return mlir::failure();

    for (auto [src, dst] :
         llvm::zip(op.getCaseRegions(), scfSwitch.getCaseRegions())) {
      rewriter.inlineRegionBefore(src, dst, dst.end());
      if (failed(rewriter.convertRegionTypes(&dst, *getTypeConverter())))
        return mlir::failure();
    }

    rewriter.replaceOp(op, scfSwitch.getResults());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// hlcf.loop + hlcf.continue + hlcf.break → scf.while
//
// scf.while state: (i1 continue_flag, break_results..., loop_carried...)
// before block: scf.condition(%continue_flag, break_results...,
//               loop_carried...)
// after block:  (break_results..., loop_carried...), the HLCF body
//   hlcf.continue %lc → scf.yield(%true, poison_breaks..., %lc)
//   hlcf.break %br    → scf.yield(%false, %br, poison_lc...)
//===----------------------------------------------------------------------===//

struct ConvertHLCFLoop : public DirectOpConversionPattern<HLCF::LoopOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(HLCF::LoopOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (hasNestedFunctionExit(op))
      return rewriter.notifyMatchFailure(
          op, "a nested `return` has no structured-control-flow form");
    // `scf.while` carries its state out through one after-region yield, so
    // only an exit terminating the loop body can reach it.
    if (Operation *exit = findExitBelowBody(op))
      return rewriter.notifyMatchFailure(exit, [&](Diagnostic &diag) {
        diag << "`" << exit->getName().stripDialect() << "` is nested inside `"
             << exit->getParentOp()->getName()
             << "`; `lower-hlcf-to-scf` only lowers a loop whose "
                "`continue`/`break` terminates the loop body itself";
      });

    Location loc = op.getLoc();

    SmallVector<Type> breakResultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                breakResultTypes)))
      return mlir::failure();
    SmallVector<Type> lcTypes;
    if (failed(getTypeConverter()->convertTypes(op.getOperands().getTypes(),
                                                lcTypes)))
      return mlir::failure();

    Type i1 = rewriter.getI1Type();
    Value constTrue = arith::ConstantIntOp::create(rewriter, loc, i1, 1LL);
    Value constFalse = arith::ConstantIntOp::create(rewriter, loc, i1, 0LL);

    SmallVector<Value> initVals;
    initVals.push_back(constTrue);
    for (Type t : breakResultTypes)
      initVals.push_back(ub::PoisonOp::create(rewriter, loc, t));
    for (Value lc : adaptor.getOperands())
      initVals.push_back(lc);

    SmallVector<Type> stateTypes;
    stateTypes.push_back(i1);
    stateTypes.append(breakResultTypes.begin(), breakResultTypes.end());
    stateTypes.append(lcTypes.begin(), lcTypes.end());

    SmallVector<Type> whileResultTypes(stateTypes.begin() + 1,
                                       stateTypes.end());

    auto whileOp =
        scf::WhileOp::create(rewriter, loc, whileResultTypes, initVals);

    Block *beforeBlock =
        rewriter.createBlock(&whileOp.getBefore(), {}, stateTypes,
                             SmallVector<Location>(stateTypes.size(), loc));
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToEnd(beforeBlock);
      Value continueFlag = beforeBlock->getArgument(0);
      SmallVector<Value> condVals(beforeBlock->args_begin() + 1,
                                  beforeBlock->args_end());
      scf::ConditionOp::create(rewriter, loc, continueFlag, condVals);
    }

    unsigned numBreakResults = breakResultTypes.size();
    SmallVector<Type> afterArgTypes(stateTypes.begin() + 1, stateTypes.end());
    Block *afterEntry =
        rewriter.createBlock(&whileOp.getAfter(), {}, afterArgTypes,
                             SmallVector<Location>(afterArgTypes.size(), loc));

    // HLCF body block args are (loop_carried...) and map to the last N args
    // of afterEntry (first N are break_results...).
    rewriter.mergeBlocks(
        &op.getBody().front(), afterEntry,
        bridgedForMerge(
            rewriter, loc, op.getBody().front(), afterEntry,
            afterEntry->getArguments().drop_front(numBreakResults)));

    if (failed(rewriter.convertRegionTypes(&whileOp.getAfter(),
                                           *getTypeConverter())))
      return mlir::failure();

    // The body's own terminator becomes the after-region yield.
    for (Block &block : whileOp.getAfter()) {
      Operation *termOp = block.getTerminator();
      if (!isa<HLCF::ContinueOp, HLCF::BreakOp>(termOp))
        continue;
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(termOp);
      SmallVector<Value> yieldVals;
      if (auto cont = dyn_cast<HLCF::ContinueOp>(termOp)) {
        yieldVals.push_back(constTrue);
        for (Type t : breakResultTypes)
          yieldVals.push_back(ub::PoisonOp::create(rewriter, loc, t));
        appendConverted(rewriter, loc, cont.getOperands(), lcTypes, yieldVals);
      } else {
        auto brk = cast<HLCF::BreakOp>(termOp);
        yieldVals.push_back(constFalse);
        appendConverted(rewriter, loc, brk.getOperands(), breakResultTypes,
                        yieldVals);
        for (Type t : lcTypes)
          yieldVals.push_back(ub::PoisonOp::create(rewriter, loc, t));
      }
      rewriter.replaceOpWithNewOp<scf::YieldOp>(termOp, yieldVals);
    }

    // scf.while results: (break_results..., loop_carried_at_exit...)
    // hlcf.loop results: (break_results...) only
    ValueRange whileResults = whileOp.getResults();
    rewriter.replaceOp(op, whileResults.take_front(breakResultTypes.size()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct LowerHLCFToSCFPass
    : public M::KGEN::impl::LowerHLCFToSCFBase<LowerHLCFToSCFPass> {
  using LowerHLCFToSCFBase::LowerHLCFToSCFBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    // Reported here because a pattern's `notifyMatchFailure` never reaches
    // the user, leaving only the driver's "failed to legalize".
    mlir::WalkResult unsupported = module.walk([](HLCF::LoopOp loop) {
      Operation *exit = findExitBelowBody(loop);
      if (!exit)
        return mlir::WalkResult::advance();
      loop.emitError("`lower-hlcf-to-scf` lowers an `hlcf.loop` only when its "
                     "`continue`/`break` terminates the loop body; this one "
                     "exits from inside a nested region")
              .attachNote(exit->getLoc())
          << "`" << exit->getName().stripDialect() << "` nested in `"
          << exit->getParentOp()->getName() << "` here";
      return mlir::WalkResult::interrupt();
    });
    if (unsupported.wasInterrupted())
      return signalPassFailure();

    POPScalarTypeConverter typeConverter(ctx);

    RewritePatternSet patterns(ctx);
    if (forOnly)
      populateHLCFForToSCFConversionPatterns(typeConverter, patterns);
    else
      populateHLCFToSCFConversionPatterns(typeConverter, patterns);

    // Every HLCF op must convert except the function terminators, which go
    // with the function conversion.
    auto mustConvert = [](Operation *op) {
      return isa_and_present<HLCF::HLCFDialect>(op->getDialect()) &&
             !isa<HLCF::ReturnOp, HLCF::UnreachableOp>(op);
    };

    if (failed(M::applyDirectConversion(module, &typeConverter,
                                        std::move(patterns),
                                        /*skipOp=*/nullptr, mustConvert)))
      signalPassFailure();
  }
};

} // namespace

void M::KGEN::populateHLCFForToSCFConversionPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ConvertHLCFFor, ConvertHLCFForYield>(converter,
                                                    patterns.getContext());
}

void M::KGEN::populateHLCFToSCFConversionPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns) {
  populateHLCFForToSCFConversionPatterns(converter, patterns);
  patterns
      .add<ConvertHLCFIf, ConvertHLCFYield, ConvertHLCFSwitch, ConvertHLCFLoop>(
          converter, patterns.getContext());
}
