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
using namespace KGEN;
using namespace POP;
using namespace ZAP;

namespace {

//===----------------------------------------------------------------------===//
// ConvertZAPBufferStackAllocation
//===----------------------------------------------------------------------===//

struct ConvertZAPBufferStackAllocation
    : mlir::OpRewritePattern<BufferStackAllocationOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferStackAllocationOp op,
                                PatternRewriter &rewriter) const override {
    auto type = op.getType().cast<BufferType>();
    Value ptr = rewriter.create<StackAllocationOp>(
        op.getLoc(), getPointerOfSameDType(type), type.getSize());
    rewriter.replaceOpWithNewOp<BufferConstructOp>(op, type, ptr, Value(),
                                                   Value());
    return success();
  }
};

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

//===----------------------------------------------------------------------===//
// ConvertZAPSIMDLoad
//===----------------------------------------------------------------------===//

struct ConvertZAPSIMDLoad : mlir::OpRewritePattern<SIMDLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SIMDLoadOp op,
                                PatternRewriter &rewriter) const override {
    Value base = rewriter.create<BufferAddressOp>(op.getLoc(), op.getBuffer());
    Value ptr = rewriter.create<OffsetOp>(op.getLoc(), base, op.getPosition());
    Value bitcastPtr = rewriter.create<BitcastOp>(
        op.getLoc(), PointerType::get(TypeConstantAttr::get(op.getType())),
        ptr);
    rewriter.replaceOpWithNewOp<LoadOp>(op, bitcastPtr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPSIMDStore
//===----------------------------------------------------------------------===//

struct ConvertZAPSIMDStore : mlir::OpRewritePattern<SIMDStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SIMDStoreOp op,
                                PatternRewriter &rewriter) const override {
    Value base = rewriter.create<BufferAddressOp>(op.getLoc(), op.getBuffer());
    Value ptr = rewriter.create<OffsetOp>(op.getLoc(), base, op.getPosition());
    Value bitcastPtr = rewriter.create<BitcastOp>(
        op.getLoc(),
        PointerType::get(TypeConstantAttr::get(op.getValue().getType())), ptr);
    rewriter.replaceOpWithNewOp<StoreOp>(op, op.getValue(), bitcastPtr);
    return failure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateZAPToPOPPatterns(RewritePatternSet &patterns) {
  patterns
      .insert<ConvertZAPBufferLoad, ConvertZAPBufferStackAllocation,
              ConvertZAPBufferStore, ConvertZAPSIMDLoad, ConvertZAPSIMDStore>(
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
