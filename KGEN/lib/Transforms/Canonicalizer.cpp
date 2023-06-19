//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/POPDialect/POPOps.h"
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

    owningPatterns.insert<IfToSelect>(context);

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
