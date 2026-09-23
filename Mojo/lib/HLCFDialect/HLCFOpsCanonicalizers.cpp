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

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;
using namespace M::KGEN;
using namespace HLCF;

/// Erase all operations following the given OP in its parent region. The OP
/// itself does not get deleted.
static void eraseOpsAfter(PatternRewriter &rewriter, Operation *op) {
  Block *toErase =
      rewriter.splitBlock(op->getBlock(), op->getNextNode()->getIterator());
  rewriter.eraseBlock(toErase);
}

/// Replace the given op with a region. If the region ends with YieldOp then
/// uses of the results of the original op will be replaced with the
/// corresponding yielded values. Otherwise, the region must be ending with a
/// Return or a similar terminator - in that case we erase all the ops after the
/// original op as dead code.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-block region");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  if (isa<YieldOp>(terminator)) {
    // If the op block ends with yield, we rewire the values in the remaining of
    // the parent block to use the yielded values.
    rewriter.replaceOp(op, terminator->getOperands());
    rewriter.eraseOp(terminator);
  } else {
    // Delete all ops after the op - the block in the op ends with a terminator
    // that renders the remaining of the parent block dead.
    eraseOpsAfter(rewriter, op);
    rewriter.eraseOp(op);
  }
}

/// Collect body regions of an IfOp: the first then, each elif then (odd
/// indices of `$elifRegions`), and the final else. Condition regions are
/// excluded.
static void getIfBodyRegions(IfOp op, SmallVectorImpl<Region *> &bodies) {
  bodies.push_back(&op.getThenRegion());
  for (unsigned i = 1, e = op.getElifRegions().size(); i < e; i += 2)
    bodies.push_back(&op.getElifRegions()[i]);
  bodies.push_back(&op.getElseRegion());
}

/// Move every region of `src` into the corresponding empty region of `dst`
/// (same number of elif regions required).
static void moveIfRegions(PatternRewriter &rewriter, IfOp src, IfOp dst) {
  assert(src.getElifRegions().size() == dst.getElifRegions().size());
  rewriter.inlineRegionBefore(src.getThenRegion(), dst.getThenRegion(),
                              dst.getThenRegion().begin());
  rewriter.inlineRegionBefore(src.getElseRegion(), dst.getElseRegion(),
                              dst.getElseRegion().begin());
  for (auto [srcRegion, dstRegion] :
       llvm::zip(src.getElifRegions(), dst.getElifRegions()))
    rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.begin());
}

/// When the first condition of a multi-arm if is statically false, drop the
/// dead then arm and promote the first elif arm to be the new if.
static LogicalResult dropDeadThenPromoteFirstElif(IfOp op,
                                                  PatternRewriter &rewriter) {
  assert(!op.getElifRegions().empty() &&
         "expected at least one elif (cond, then) pair");
  Region &condRegion = op.getElifRegions()[0];
  Region &thenRegion = op.getElifRegions()[1];
  auto yield = dyn_cast<IfElifCondYieldOp>(condRegion.front().getTerminator());
  // Only promote when the condition region is a trivial yield of a condition
  // with no mem2reg carry-over values (those become block args on then/else).
  if (!yield || &condRegion.front().front() != yield ||
      !yield.getValues().empty())
    return failure();

  unsigned remainingElifs = op.getElifRegions().size() - 2;
  auto newOp = IfOp::create(rewriter, op.getLoc(), op.getResultTypes(),
                            yield.getCond(), remainingElifs);
  rewriter.inlineRegionBefore(thenRegion, newOp.getThenRegion(),
                              newOp.getThenRegion().begin());
  rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                              newOp.getElseRegion().begin());
  for (unsigned i = 0; i != remainingElifs; ++i)
    rewriter.inlineRegionBefore(op.getElifRegions()[i + 2],
                                newOp.getElifRegions()[i],
                                newOp.getElifRegions()[i].begin());
  rewriter.replaceOp(op, newOp.getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// IfOp canonicalizations
//===----------------------------------------------------------------------===//

namespace {
/// If every body arm has just a YieldOp and corresponding results agree,
/// replace those results (and erase the if when all results fold away).
///
///   Before:
///      %a, %b = hlcf.if %cond {
///        hlcf.yield %c, %d
///      } else {
///        hlcf.yield %c, %d
///      }
///      return %a, %b
///
///   After:
///      return %c, %d
struct HoistYieldResults : public OpRewritePattern<IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Region *, 4> bodies;
    getIfBodyRegions(op, bodies);

    SmallVector<YieldOp, 4> yields;
    yields.reserve(bodies.size());
    for (Region *body : bodies) {
      auto yield = dyn_cast<YieldOp>(body->front().getTerminator());
      if (!yield || &body->front().front() != yield ||
          yield->getNumOperands() != op.getNumResults())
        return failure();
      yields.push_back(yield);
    }

    bool changed = false;
    bool allChanged = true;
    for (unsigned resIdx = 0, e = op.getNumResults(); resIdx != e; ++resIdx) {
      Value res = op.getResult(resIdx);
      Value first = yields.front()->getOperand(resIdx);
      bool allSame = llvm::all_of(
          yields, [&](YieldOp y) { return y->getOperand(resIdx) == first; });

      // Two-arm special case: `if cond { yield true } else { yield false }`
      // folds to `cond`.
      if (yields.size() == 2 && res.getType() == op.getCond().getType()) {
        KGEN::SIMDAttr trueCond, falseCond;
        if (matchPattern(yields[0]->getOperand(resIdx),
                         m_Constant(&trueCond)) &&
            matchPattern(yields[1]->getOperand(resIdx),
                         m_Constant(&falseCond)) &&
            trueCond.getAsBool() == true && falseCond.getAsBool() == false) {
          rewriter.replaceAllUsesWith(res, op.getCond());
          changed = true;
          continue;
        }
      }

      if (allSame) {
        rewriter.replaceAllUsesWith(res, first);
        changed = true;
      } else {
        allChanged = false;
      }
    }
    // Safe: every body arm is only a yield.
    if (allChanged) {
      rewriter.eraseOp(op);
      changed = true;
    }
    return changed ? success() : failure();
  }
};

/// If the first condition is known at compile time, keep only the live arm(s).
/// True → replace with the then region. False with no elif → replace with the
/// else region. False with elifs → drop the dead then and promote the first
/// elif arm to a new if (when its condition region is a trivial yield).
struct RemoveStaticCondition : public OpRewritePattern<IfOp> {
  RemoveStaticCondition(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    KGEN::SIMDAttr condition;
    if (!matchPattern(op.getCond(), m_Constant(&condition)))
      return failure();

    if (condition.getAsBool()) {
      replaceOpWithRegion(rewriter, op, op.getThenRegion());
      return success();
    }

    if (op.getElifRegions().empty()) {
      replaceOpWithRegion(rewriter, op, op.getElseRegion());
      return success();
    }
    return dropDeadThenPromoteFirstElif(op, rewriter);
  }
};

/// If every body arm ends with the same return/break/continue, replace those
/// terminators with yields and insert one terminator after the if.
///
/// Before:                    After:
/// {                          {
///   ...                        ...
///   if %cond {                 %x = if %cond {
///      A                          A
///      return %a                  yield %a
///   } else {                   } else {
///      B                          B
///      return %b                  yield %b
///   }                          }
///   C                          return %x
/// }                          }
template <typename TerminatorOpT>
struct HoistUnconditionalReturn : public OpRewritePattern<IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Region *, 4> bodies;
    getIfBodyRegions(op, bodies);

    SmallVector<TerminatorOpT, 4> terms;
    terms.reserve(bodies.size());
    for (Region *body : bodies) {
      auto term = dyn_cast<TerminatorOpT>(body->front().getTerminator());
      if (!term)
        return failure();
      terms.push_back(term);
    }

    // All arms become yields of the new op's results; arities must match.
    TypeRange operandTypes = terms.front()->getOperandTypes();
    if (llvm::any_of(terms, [&](TerminatorOpT term) {
          return term->getOperandTypes() != operandTypes;
        }))
      return failure();

    if constexpr (!std::is_same_v<TerminatorOpT, HLCF::ReturnOp>) {
      // Ensure all terminators branch to the same labeled loop.
      auto label = terms.front().getLabelAttr();
      if (llvm::any_of(terms, [&](TerminatorOpT term) {
            return term.getLabelAttr() != label;
          }))
        return failure();
    }

    auto newOp = IfOp::create(rewriter, op.getLoc(), operandTypes, op.getCond(),
                              op.getElifRegions().size());
    TerminatorOpT::create(
        rewriter, op.getLoc(), TypeRange(), newOp->getResults(),
        terms.front().getProperties(),
        llvm::to_vector(terms.front()->getDiscardableAttrs()));

    moveIfRegions(rewriter, op, newOp);

    // Replace each body terminator with a yield of its operands.
    SmallVector<Region *, 4> newBodies;
    getIfBodyRegions(newOp, newBodies);
    for (Region *body : newBodies) {
      Operation *term = body->front().getTerminator();
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<YieldOp>(term, term->getOperands());
    }

    eraseOpsAfter(rewriter, op);
    rewriter.eraseOp(op);
    return success();
  }
};

/// If some body arms exit with Return/Break and the others Yield, pull the
/// code after the if into each yield arm and hoist a single exit after the if.
///
/// Before:                    After:
/// {                          {
///   ...                        ...
///   %x = if %cond {            %x = if %cond {
///      %a = A                     %a = A
///      return %a                  yield %a
///   } else {                   } else {
///      %b = B                     %b = B
///      yield %b                   %t = C(%b)
///                                 yield %t
///   }                          }
///   %t = C(%x)                 return %x
///   return %t
/// }                          }
struct HoistConditionalReturn : public OpRewritePattern<IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Region *, 4> bodies;
    getIfBodyRegions(op, bodies);

    SmallVector<Operation *, 4> yieldTerms;
    SmallVector<Operation *, 4> exitTerms;
    for (Region *body : bodies) {
      Operation *term = body->front().getTerminator();
      if (isa<YieldOp>(term))
        yieldTerms.push_back(term);
      else if (isa<HLCF::ReturnOp, BreakOp>(term))
        exitTerms.push_back(term);
      else
        return rewriter.notifyMatchFailure(
            op, "Body arm does not end with Yield/Return/Break");
    }

    if (exitTerms.empty())
      return rewriter.notifyMatchFailure(
          op, "None of the branches ends with Return/Break");
    if (yieldTerms.empty())
      return rewriter.notifyMatchFailure(
          op, "None of the branches ends with Yield");

    // All exiting arms must use the same kind of terminator with matching
    // operands/labels.
    Operation *sampleExit = exitTerms.front();
    if (llvm::any_of(exitTerms, [&](Operation *term) {
          return term->getName() != sampleExit->getName() ||
                 term->getOperandTypes() != sampleExit->getOperandTypes();
        }))
      return rewriter.notifyMatchFailure(
          op, "Exiting arms have mismatched terminators");
    if (auto br = dyn_cast<BreakOp>(sampleExit)) {
      if (llvm::any_of(exitTerms, [&](Operation *term) {
            return cast<BreakOp>(term).getLabelAttr() != br.getLabelAttr();
          }))
        return rewriter.notifyMatchFailure(op,
                                           "Break arms target different loops");
    }

    // Walk out through nested 2-arm if yields to find the enclosing
    // return/break that the yield arm(s) ultimately feed.
    SmallVector<Value> parentBlockTermOperands;
    Operation *actualParentTermOp = nullptr;
    std::function<void(Operation *)> findParentTermOp;
    findParentTermOp = [&](Operation *parentTerm) {
      if (isa<HLCF::ReturnOp, BreakOp>(parentTerm)) {
        actualParentTermOp = parentTerm;
        parentBlockTermOperands = parentTerm->getOperands();
        return;
      }

      auto yield = dyn_cast<YieldOp>(parentTerm);
      auto parentIf = yield ? dyn_cast<IfOp>(yield->getParentOp()) : IfOp();
      // Only climb through simple 2-arm ifs (no extra elif arms).
      if (!yield || !parentIf || !parentIf.getElifRegions().empty())
        return;
      Operation &termAfterIf = *std::next(Block::iterator(parentIf));
      Operation *blockTerm = parentIf->getBlock()->getTerminator();
      if (!isa<HLCF::ReturnOp, BreakOp, YieldOp>(termAfterIf) &&
          parentTerm != yieldTerms.front())
        return;
      findParentTermOp(blockTerm);
      if (!actualParentTermOp)
        return;

      for (auto &retVal : parentBlockTermOperands) {
        OpResult retRes = dyn_cast<OpResult>(retVal);
        if (!retRes || retRes.getOwner() != parentIf)
          continue;
        if (retRes.getResultNumber() >= yield->getNumOperands()) {
          actualParentTermOp = nullptr;
          return;
        }
        retVal = yield.getOperand(retRes.getResultNumber());
      }
    };
    findParentTermOp(yieldTerms.front());

    if (!actualParentTermOp)
      return rewriter.notifyMatchFailure(
          op, "Parent block doesn't end with Return/Break");

    // Nested climbing remaps through a specific yield arm's values; that does
    // not generalize to multiple yield arms.
    if (yieldTerms.size() > 1 &&
        actualParentTermOp != op->getBlock()->getTerminator())
      return rewriter.notifyMatchFailure(
          op, "nested multi-arm conditional hoist is unsupported");

    if (isa<HLCF::ReturnOp>(actualParentTermOp)) {
      if (!isa<HLCF::ReturnOp>(sampleExit))
        return rewriter.notifyMatchFailure(
            op, "Parent block is Return, but exiting terminator is Break");
    } else {
      assert(isa<BreakOp>(actualParentTermOp));
      if (!isa<BreakOp>(sampleExit))
        return rewriter.notifyMatchFailure(
            op, "Parent block is Break, but exiting terminator is Return");
      if (cast<BreakOp>(actualParentTermOp).getLabelAttr() !=
          cast<BreakOp>(sampleExit).getLabelAttr())
        return rewriter.notifyMatchFailure(
            op, "Break in the parent block's target is different from exiting "
                "terminator break's target");
    }

    if (sampleExit->getNumOperands() != actualParentTermOp->getNumOperands() ||
        !llvm::equal(sampleExit->getOperandTypes(),
                     actualParentTermOp->getOperandTypes()))
      return rewriter.notifyMatchFailure(
          op, "Exiting terminator and parent return/break have different "
              "operand types");
    for (Operation *yieldTerm : yieldTerms) {
      if (yieldTerm->getNumOperands() != op.getNumResults())
        return rewriter.notifyMatchFailure(
            op, "Yield operand count doesn't match if results");
    }
    if (parentBlockTermOperands.size() != sampleExit->getNumOperands())
      return rewriter.notifyMatchFailure(
          op, "Remapped parent terminator operands don't match exiting "
              "terminator");

    // Snapshot ops after the if before mutating the block.
    SmallVector<Operation *, 8> remainderOps;
    for (Operation *o = op->getNextNode(); o; o = o->getNextNode())
      remainderOps.push_back(o);

    auto newOp = IfOp::create(rewriter, op.getLoc(),
                              actualParentTermOp->getOperandTypes(),
                              op.getCond(), op.getElifRegions().size());
    moveIfRegions(rewriter, op, newOp);

    SmallVector<Region *, 4> newBodies;
    getIfBodyRegions(newOp, newBodies);
    SmallVector<Operation *, 4> newYieldTerms;
    SmallVector<Operation *, 4> newExitTerms;
    for (Region *body : newBodies) {
      Operation *term = body->front().getTerminator();
      if (isa<YieldOp>(term))
        newYieldTerms.push_back(term);
      else
        newExitTerms.push_back(term);
    }

    // Clone the trailing code into every yield arm, remapping if results to
    // that arm's yield operands, then turn the cloned terminator into a yield.
    for (Operation *yieldTerm : newYieldTerms) {
      rewriter.setInsertionPoint(yieldTerm);
      IRMapping mapping;
      for (auto [idx, val] : llvm::enumerate(op->getResults()))
        mapping.map(val, yieldTerm->getOperand(idx));

      Operation *clonedTerm = nullptr;
      for (Operation *origOp : remainderOps)
        clonedTerm = rewriter.clone(*origOp, mapping);

      SmallVector<Value> yieldOperands;
      yieldOperands.reserve(parentBlockTermOperands.size());
      for (Value v : parentBlockTermOperands) {
        if (auto res = dyn_cast<OpResult>(v); res && res.getOwner() == op)
          yieldOperands.push_back(yieldTerm->getOperand(res.getResultNumber()));
        else
          yieldOperands.push_back(mapping.lookupOrDefault(v));
      }
      rewriter.eraseOp(yieldTerm);
      rewriter.replaceOpWithNewOp<YieldOp>(clonedTerm, yieldOperands);
    }

    for (Operation *o : llvm::reverse(remainderOps))
      rewriter.eraseOp(o);

    rewriter.setInsertionPointAfter(newOp);
    if (auto br = dyn_cast<BreakOp>(sampleExit)) {
      BreakOp::create(rewriter, op.getLoc(), newOp->getResults(),
                      br.getLabelAttr());
    } else {
      HLCF::ReturnOp::create(rewriter, op.getLoc(), newOp->getResults());
    }

    for (Operation *exitTerm : newExitTerms) {
      rewriter.setInsertionPoint(exitTerm);
      rewriter.replaceOpWithNewOp<YieldOp>(exitTerm, exitTerm->getOperands());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Remove unused results of the if and the matching operands of every body
/// yield (then, elif thens, and else).
struct RemoveUnusedResults : public OpRewritePattern<IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op, PatternRewriter &b) const override {
    SmallVector<Region *, 4> bodies;
    getIfBodyRegions(op, bodies);

    SmallVector<YieldOp, 4> yields;
    for (Region *body : bodies) {
      auto yield = dyn_cast<YieldOp>(body->front().getTerminator());
      if (yield && yield->getNumOperands() != op.getNumResults())
        return failure();
      if (yield)
        yields.push_back(yield);
    }

    llvm::BitVector unused(op.getNumResults());
    SmallVector<Value> toReplace;
    for (auto [i, result] : llvm::enumerate(op.getResults())) {
      if (result.use_empty())
        unused.set(i);
      else
        toReplace.push_back(result);
    }

    if (unused.none())
      return b.notifyMatchFailure(op.getLoc(), "all results have uses");

    for (YieldOp yield : yields)
      b.modifyOpInPlace(yield, [&] { yield->eraseOperands(unused); });

    auto newOp = IfOp::create(b, op.getLoc(), TypeRange(ValueRange(toReplace)),
                              op.getCond(), op.getElifRegions().size());
    b.replaceAllUsesWith(toReplace, newOp.getResults());
    moveIfRegions(b, op, newOp);
    b.eraseOp(op);
    return success();
  }
};
} // namespace

void IfOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *ctx) {
  results.add<RemoveStaticCondition, HoistUnconditionalReturn<HLCF::ReturnOp>,
              HoistUnconditionalReturn<HLCF::BreakOp>,
              HoistUnconditionalReturn<HLCF::ContinueOp>,
              HoistConditionalReturn, HoistYieldResults, RemoveUnusedResults>(
      ctx);
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

namespace {
/// If the only operation in LoopOp is BreakOp, delete the loop.  Depending on
/// whether the target of the BreakOp is this or outer loop, we might have to
/// keep or delete it.
struct RemoveDeadLoop : OpRewritePattern<LoopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp op, PatternRewriter &b) const override {
    Block &body = op.getBody().front();
    if (auto br = dyn_cast<BreakOp>(body.getOperations().front())) {
      StringAttr label = br.getLabelAttr();
      if (!label || label == op.getLabelAttr()) {
        SmallVector<Value> operandsToReplace = br.getOperands();
        for (auto [idx, value] : llvm::enumerate(operandsToReplace)) {
          if (auto arg = dyn_cast<BlockArgument>(value)) {
            if (arg.getOwner() != &op.getRegion().front())
              continue;
            // If the break's operand is a block argument of the loop op
            // that is about to be erased, use the loop operand instead.
            operandsToReplace[idx] = op.getOperand(arg.getArgNumber());
          }
        }
        b.replaceOp(op, operandsToReplace);
      } else {
        b.inlineBlockBefore(&body, op);
        eraseOpsAfter(b, op);
        b.eraseOp(op);
      }
      return success();
    }

    return failure();
  }
};

/// Remove unused results from a loop. This requires traversing the body to find
/// matching `break` operations, but the cost is paid only when there is a
/// match.
struct RemoveUnusedLoopResults : OpRewritePattern<LoopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp loop,
                                PatternRewriter &b) const override {
    llvm::BitVector unused(loop.getNumResults());
    SmallVector<Value> toReplace;
    for (auto [i, result] : llvm::enumerate(loop.getResults())) {
      if (result.use_empty())
        unused.set(i);
      else
        toReplace.push_back(result);
    }

    if (unused.none())
      return b.notifyMatchFailure(loop.getLoc(), "all results have uses");

    // Find all matching break operations.
    StringAttr label = loop.getLabelAttr();
    loop.getBody().walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      // Walk over loops with the same label.
      if (auto inner = dyn_cast<LoopOp>(op);
          inner && inner.getLabelAttr() == label)
        return WalkResult::skip();

      // If this is a matching break, remove the unused operands.
      if (auto breakOp = dyn_cast<BreakOp>(op);
          breakOp && getParentLoop(breakOp, breakOp.getLabelAttr()) == loop)
        b.modifyOpInPlace(breakOp, [&] { breakOp->eraseOperands(unused); });

      return WalkResult::advance();
    });

    auto newLoop =
        LoopOp::create(b, loop.getLoc(), TypeRange(ValueRange(toReplace)),
                       loop.getOperands(), label, loop.getUnrollLevelAttr());
    b.replaceAllUsesWith(toReplace, newLoop.getResults());
    b.inlineRegionBefore(loop.getBody(), newLoop.getBody(),
                         newLoop.getBody().begin());
    b.eraseOp(loop);
    return success();
  }
};

/// Remove loop arguments that are unused.
struct RemoveUnusedLoopArgs : OpRewritePattern<LoopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp loop,
                                PatternRewriter &b) const override {
    llvm::BitVector unused(loop.getNumOperands());
    for (BlockArgument arg : loop.getBody().getArguments())
      if (arg.use_empty())
        unused.set(arg.getArgNumber());
    if (unused.none())
      return b.notifyMatchFailure(loop.getLoc(), "no unused arguments");

    b.modifyOpInPlace(loop, [&] {
      loop->eraseOperands(unused);
      loop.getBody().front().eraseArguments(unused);
    });

    // Find all matching continue operations.
    StringAttr label = loop.getLabelAttr();
    loop.getBody().walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      // Walk over loops with the same label.
      if (auto inner = dyn_cast<LoopOp>(op);
          inner && inner.getLabelAttr() == label)
        return WalkResult::skip();

      // If this is a matching break, remove the unused operands.
      if (auto cont = dyn_cast<ContinueOp>(op);
          cont && getParentLoop(cont, cont.getLabelAttr()) == loop)
        b.modifyOpInPlace(cont, [&] { cont->eraseOperands(unused); });

      return WalkResult::advance();
    });

    return success();
  }
};
} // namespace

void LoopOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *ctx) {
  results.insert<RemoveDeadLoop, RemoveUnusedLoopResults, RemoveUnusedLoopArgs>(
      ctx);
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

static bool isPureOrReadOnly(Operation &op) {
  auto itf = dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!itf)
    return false;
  SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  itf.getEffects(effects);
  if (effects.empty())
    return true;
  return llvm::all_of(effects, [](auto &e) {
    return isa<mlir::MemoryEffects::Read>(e.getEffect());
  });
}

namespace {
/// Scan the body of a loop with no results up to a small number of consecutive
/// ops, checking if they are all pure or readonly. If this is the case, we know
/// the loop is overall a no-op.
struct RemoveNoopLoop : OpRewritePattern<ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op, PatternRewriter &b) const override {
    if (op.getNumResults())
      return b.notifyMatchFailure(op.getLoc(), "loop has results");

    constexpr unsigned numToScan = 5;
    for (auto [idx, op] :
         llvm::enumerate(op.getBody().front().without_terminator())) {
      if (idx > numToScan)
        return b.notifyMatchFailure(op.getLoc(), "body is too large");
      if (op.getNumRegions())
        return b.notifyMatchFailure(op.getLoc(), "body op with regions");
      if (!isPureOrReadOnly(op))
        return b.notifyMatchFailure(op.getLoc(), "not a pure or readonly op");
    }
    b.eraseOp(op);
    return success();
  }
};
} // namespace

void ForOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *ctx) {
  results.insert<RemoveNoopLoop>(ctx);
}

//===----------------------------------------------------------------------===//
// ComptimeIfOp
//===----------------------------------------------------------------------===//

LogicalResult ComptimeIfOp::canonicalize(ComptimeIfOp op, PatternRewriter &b) {
  // Multi-arm comptime.if is not folded here; keep the existing binary-arm
  // patterns only (do not synthesize elif regions).
  if (!op.getElifRegions().empty())
    return b.notifyMatchFailure(op.getLoc(),
                                "multi-arm comptime.if is not canonicalized");

  Block &ifBranch = op->getRegion(0).front();
  Block &elseBranch = op->getRegion(1).front();
  Operation *ifTerm = ifBranch.getTerminator();
  Operation *elseTerm = elseBranch.getTerminator();

  // Simple patterns to handle the case of branches containing just terminator
  // ops.
  if (ifTerm == &ifBranch.front() && elseTerm == &elseBranch.front() &&
      op->getNumResults() == 0) {
    // If both sides are yielding, we can delete the op.
    if (isa<ComptimeYieldOp>(ifTerm) && isa<ComptimeYieldOp>(elseTerm)) {
      b.eraseOp(op);
      return success();
    }

    // If one branch yields and another breaks we can delete the op if the op is
    // immediately preceding another break. The terminators can't have any
    // returns.
    if (ifTerm->getNumOperands() == 0 && elseTerm->getNumOperands() == 0 &&
        isa<ComptimeYieldOp, HLCF::BreakOp>(ifTerm) &&
        isa<ComptimeYieldOp, HLCF::BreakOp>(elseTerm) &&
        isa<HLCF::BreakOp>(op->getNextNode())) {
      b.eraseOp(op);
      return success();
    }
  }

  auto condAttr = sugarDynCast<SIMDAttr>(op.getCond());
  if (!condAttr)
    return b.notifyMatchFailure(op.getLoc(), "condition is not a constant");
  bool condValue = condAttr.getAsBool();

  // We can't fold away the op entirely, because it defines a parameter scope
  // and this could create param decl conflicts. Instead, purge the dead region
  // and insert a `hlcf.unreachable`.
  Block &deadBlock = op->getRegion(condValue).front();

  // Don't match again if the dead block is already purged.
  if (isa<UnreachableOp>(deadBlock.front()))
    return b.notifyMatchFailure(op.getLoc(), "dead block already purged");

  // Hoist all the non parameter defining ops out of the live region.
  Block &liveBlock = op->getRegion(!condValue).front();
  while (!liveBlock.front().hasTrait<OpTrait::IsTerminator>()) {
    // Stop if we hit an operation defining a parameter. We don't hoist these as
    // the parameter regions could conflict.
    if (auto paramOp = dyn_cast<ParamOpInterface>(liveBlock.front())) {
      bool hasParam = false;
      paramOp.walkDeclarations([&](ParamDeclAttr attr) { hasParam = true; });
      if (hasParam)
        break;
    }

    // Otherwise, hoist the operation above the 'if'.
    b.moveOpBefore(&liveBlock.front(), op);
  }

  // If we got down to a terminator that we can handle, eliminate the 'if'.
  Operation &liveFront = liveBlock.front();
  // If the live block is now trivial, we can remove the whole
  // operation. Replace the results with the operands to the yield.
  if (auto yield = dyn_cast<ComptimeYieldOp>(liveFront)) {
    b.replaceOp(op, yield.getOperands());
    return success();
  }

  // If we are ending control flow we can hoist it out but we have to delete
  // all following ops to retain legality.
  if (isa<HLCF::UnreachableOp, HLCF::BreakOp, HLCF::ContinueOp>(liveFront)) {
    Block *block = op->getBlock();
    // Delete things bottom-up so we delete uses before defs.
    while (&block->back() != op)
      b.eraseOp(&block->back());
    // Move the terminator out of the 'if' and remove the 'if'.
    b.moveOpBefore(&liveFront, op);
    b.eraseOp(op);
    return success();
  }

  // Otherwise, we have a parameter defining op (which we need the scope for)
  // or control flow we don't know about.
  for (Operation &subOp : llvm::make_early_inc_range(llvm::reverse(deadBlock)))
    b.eraseOp(&subOp);
  b.setInsertionPointToStart(&deadBlock);
  UnreachableOp::create(b, op.getLoc());
  return success();
}
