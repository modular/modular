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
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ZAPDialect/ZAPOps.h"
#include "Support/IndexDialect/IndexOps.h"
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
        op.getLoc(), PointerType::get(ScalarType::get(type.getDType())),
        type.getSize());
    rewriter.replaceOpWithNewOp<BufferConstructOp>(op, type, ptr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferConstant
//===----------------------------------------------------------------------===//

/// Create a buffer constant with the given values.
static Value createBufferConstant(PatternRewriter &rewriter, Location loc,
                                  BufferType type, DenseElementsAttr values) {
  auto elType = ScalarType::get(type.getDType());
  Value global = rewriter.create<GlobalConstantOp>(
      loc, PointerType::get(ArrayType::get(type.getSize(), elType)), values);
  Value ptr = rewriter.create<BitcastOp>(loc, PointerType::get(elType), global);
  return rewriter.create<BufferConstructOp>(loc, type, ptr);
}

struct ConvertZAPBufferConstant : mlir::OpRewritePattern<BufferConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferConstantOp op,
                                PatternRewriter &rewriter) const override {
    Value buf = createBufferConstant(rewriter, op.getLoc(), op.getType(),
                                     op.getValues());
    rewriter.replaceOp(op, buf);
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
    return success();
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
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPPrint
//===----------------------------------------------------------------------===//

struct ConvertZAPPrint : mlir::OpRewritePattern<PrintOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PrintOp op,
                                PatternRewriter &rewriter) const override {
    // Lower the string into the a buffer constant. Null-terminate the string.
    SmallVector<char> fmtStr;
    fmtStr.reserve(op.getFmt().size() + 1);
    llvm::append_range(fmtStr, op.getFmt());
    fmtStr.push_back('\0');
    auto values = DenseIntElementsAttr::get(
        RankedTensorType::get(fmtStr.size(), rewriter.getIntegerType(8, true)),
        ArrayRef<char>(fmtStr.data(), fmtStr.size()));
    auto fmtType = rewriter.getType<BufferType>(fmtStr.size(), DType::si8);
    Value fmt = createBufferConstant(rewriter, op.getLoc(), fmtType, values);
    // Create the invocation to `printf`.
    SmallVector<Value> operands;
    operands.reserve(op.getNumOperands() + 1);
    operands.push_back(fmt);
    llvm::append_range(operands, op.getOperands());
    rewriter.replaceOpWithNewOp<ExternalCallOp>(
        op, TypeRange(), "printf", operands,
        TypeAttr::get(rewriter.getFunctionType(fmtType, {})));
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateZAPToPOPPatterns(RewritePatternSet &patterns) {
  patterns.insert<ConvertZAPBufferLoad, ConvertZAPBufferConstant,
                  ConvertZAPBufferStackAllocation, ConvertZAPBufferStore,
                  ConvertZAPPrint, ConvertZAPSIMDLoad, ConvertZAPSIMDStore>(
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
