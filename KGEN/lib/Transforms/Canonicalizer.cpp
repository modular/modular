//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Return true if the region's body is empty (only contains a terminator).
static bool isEmpty(Region &region) {
  assert(llvm::hasSingleElement(region));
  return llvm::hasSingleElement(region.front());
}

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Canonicalize ifs with no bodies an N results to N selects. This also removes
/// trivially dead ifs.
struct IfToSelect : public OpRewritePattern<HLCF::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(HLCF::IfOp op,
                                PatternRewriter &b) const override {
    auto thenYield = dyn_cast<HLCF::YieldOp>(op.getThenTerminator());
    auto elseYield = dyn_cast<HLCF::YieldOp>(op.getElseTerminator());
    if (!isEmpty(op.getThenRegion()) || !isEmpty(op.getElseRegion()) ||
        !thenYield || !elseYield)
      return b.notifyMatchFailure(op.getLoc(),
                                  "bodies aren't empty with 'yield'");

    // Replace each result with a 'select' of the yield operands.
    SmallVector<Value> replacements;
    for (auto [i, result] : llvm::enumerate(op.getResults())) {
      replacements.push_back(b.create<POP::SelectOp>(op.getLoc(), op.getCond(),
                                                     thenYield.getOperand(i),
                                                     elseYield.getOperand(i)));
    }

    b.replaceOp(op, replacements);
    return success();
  }
};

/// Canonicalize `!pop.scalar<index>` computations to `index` operations.
class IndexifyComparison : public OpRewritePattern<POP::CastToBuiltinOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(POP::CastToBuiltinOp op,
                                PatternRewriter &b) const override {
    if (op.getInput().getType().getResolvedDType() != KGENDType::kBool)
      return b.notifyMatchFailure(op.getLoc(), "not bool dtype");

    auto cmp = op.getInput().getDefiningOp<POP::CmpOp>();
    if (!cmp || !cmp->hasOneUse())
      return b.notifyMatchFailure(op.getLoc(),
                                  "input isn't single-use comparison");

    auto cast = cmp.getLhs().getDefiningOp<POP::CastFromBuiltinOp>();
    if (!cast || !cast->hasOneUse() ||
        cast.getResult().getType().getResolvedDType() != KGENDType::index)
      return b.notifyMatchFailure(op.getLoc(),
                                  "LHS isn't single-use cast to index dtype");

    POP::SIMDAttr rhs;
    if (!mlir::matchPattern(cmp.getRhs(), mlir::m_Constant(&rhs)))
      return b.notifyMatchFailure(op.getLoc(), "RHS isn't constant");

    auto cst = b.create<mlir::index::ConstantOp>(
        op.getLoc(), rhs.getValues().front().getIndexVal());
    b.replaceOpWithNewOp<mlir::index::CmpOp>(
        op, getIndexCmpPredicate(cmp.getPred()), cast.getInput(), cst);
    b.eraseOp(cmp);
    b.eraseOp(cast);
    return success();
  }

private:
  /// Get the equivalent index comparison predicate. POP treats the `index`
  /// dtype as signed.
  static mlir::index::IndexCmpPredicate
  getIndexCmpPredicate(POP::CmpPredicate pred) {
    switch (pred) {
    case POP::CmpPredicate::EQ:
      return mlir::index::IndexCmpPredicate::EQ;
    case POP::CmpPredicate::NE:
      return mlir::index::IndexCmpPredicate::NE;
    case POP::CmpPredicate::LT:
      return mlir::index::IndexCmpPredicate::SLT;
    case POP::CmpPredicate::GT:
      return mlir::index::IndexCmpPredicate::SGT;
    case POP::CmpPredicate::LE:
      return mlir::index::IndexCmpPredicate::SLE;
    case POP::CmpPredicate::GE:
      return mlir::index::IndexCmpPredicate::SGE;
    }
  }
};

/// Canonicalize `(i < x ? x - i : 0) > 0` to `i < x`. This is a common pattern
/// in for loop constructs.
/// TODO: Generalize this pattern?
struct SimplifyCompareSelect : OpRewritePattern<mlir::index::CmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::index::CmpOp op,
                                PatternRewriter &b) const override {
    if (op.getPred() != mlir::index::IndexCmpPredicate::SGT)
      return b.notifyMatchFailure(op.getLoc(), "predicate is not `sgt`");

    IntegerAttr cmpRhs;
    if (!mlir::matchPattern(op.getRhs(), mlir::m_Constant(&cmpRhs)) ||
        !cmpRhs.getValue().isZero())
      return b.notifyMatchFailure(op.getLoc(), "RHS is not zero");

    auto select = op.getLhs().getDefiningOp<POP::SelectOp>();
    if (!select)
      return b.notifyMatchFailure(op.getLoc(), "LHS is not a select");

    auto indexCmp = select.getCondition().getDefiningOp<mlir::index::CmpOp>();
    if (!indexCmp || indexCmp.getPred() != mlir::index::IndexCmpPredicate::SLT)
      return b.notifyMatchFailure(op.getLoc(),
                                  "select condition is not `slt` comparison");

    auto trueVal = select.getTrueValue().getDefiningOp<mlir::index::SubOp>();
    if (!trueVal || trueVal.getLhs() != indexCmp.getRhs() ||
        trueVal.getRhs() != indexCmp.getLhs())
      return b.notifyMatchFailure(op.getLoc(),
                                  "select true value is not `x - i`");

    IntegerAttr falseVal;
    if (!mlir::matchPattern(select.getFalseValue(),
                            mlir::m_Constant(&falseVal)) ||
        falseVal != cmpRhs)
      return b.notifyMatchFailure(op.getLoc(),
                                  "select false value is not zero");

    // Just replace the whole thing with `i < x`.
    b.replaceOp(op, indexCmp);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_CANONICALIZER
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct Canonicalizer : public impl::CanonicalizerBase<Canonicalizer> {
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (mlir::RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    owningPatterns
        .insert<IfToSelect, IndexifyComparison, SimplifyCompareSelect>(context);

    patterns = mlir::FrozenRewritePatternSet(std::move(owningPatterns));
    return success();
  }

  void runOnOperation() override {
    mlir::GreedyRewriteConfig config;
    config.enableRegionSimplification = false;
    (void)applyPatternsAndFoldGreedily(getOperation(), patterns, config);
  }

  mlir::FrozenRewritePatternSet patterns;
};
} // namespace
