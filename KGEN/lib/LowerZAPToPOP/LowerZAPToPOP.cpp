//===- LowerZAPToPOP.cpp --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"
#include "KGEN/MetaDialect/MetaDialect.h"
#include "KGEN/MetaDialect/MetaOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ZAPDialect/ZAPOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;
using namespace M::KGEN;

namespace {

//===----------------------------------------------------------------------===//
// ConvertZAPBufferLoad
//===----------------------------------------------------------------------===//

struct ConvertZAPBufferLoad : mlir::OpRewritePattern<BufferLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferLoadOp op,
                                PatternRewriter &rewriter) const override {
    Value base = rewriter.create<BufferAddressOp>(op.getLoc(), op.getBuffer());
    Value ptr = rewriter.create<OffsetOp>(op.getLoc(), base, op.getPosition());
    rewriter.replaceOpWithNewOp<LoadOp>(op, ptr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferStore
//===----------------------------------------------------------------------===//

struct ConvertZAPBufferStore : mlir::OpRewritePattern<BufferStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferStoreOp op,
                                PatternRewriter &rewriter) const override {
    Value base = rewriter.create<BufferAddressOp>(op.getLoc(), op.getBuffer());
    Value ptr = rewriter.create<OffsetOp>(op.getLoc(), base, op.getPosition());
    rewriter.replaceOpWithNewOp<StoreOp>(op, op.getValue(), ptr);
    return failure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateZAPToPOPPatterns(RewritePatternSet &patterns) {
  patterns.insert<ConvertZAPBufferLoad, ConvertZAPBufferStore>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct LowerZAPToPOPPass : public LowerZAPToPOPBase<LowerZAPToPOPPass> {
  void runOnOperation() override;
};
} // namespace

void LowerZAPToPOPPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateZAPToPOPPatterns(patterns);
  if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> M::KGEN::createLowerZAPToPOPPass() {
  return std::make_unique<LowerZAPToPOPPass>();
}
