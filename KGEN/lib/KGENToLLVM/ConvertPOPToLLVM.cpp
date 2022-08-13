//===- ConvertPOPToLLVM.cpp -----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MetaDialect/MetaTypeConverter.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

namespace {

//===----------------------------------------------------------------------===//
// OneToOneConversion
//===----------------------------------------------------------------------===//

/// This pattern does a one-to-one conversion of one operation to another.
template <typename FromOp, typename ToOp>
struct OneToOneConversion : public OpConversionPattern<FromOp> {
  using OpConversionPattern<FromOp>::OpConversionPattern;

  LogicalResult match(FromOp op) const override { return success(); }

  void rewrite(FromOp op, typename FromOp::Adaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ToOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getOperands(), op->getAttrs());
  }
};

//===----------------------------------------------------------------------===//
// OneToOneIntOrFloatConversion
//===----------------------------------------------------------------------===//

/// This patterns converts a scalar POP dialect operation to either an integer
/// or floating point LLVM operation one-to-one.
template <typename Op, typename IntOp, typename FloatOp>
struct OneToOneIntOrFloatConversion : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DType dtype =
        op.getType().template cast<DataTypeInterface>().resolveDType();
    Type type = this->getTypeConverter()->convertType(op.getType());
    if (dtype.isInt())
      rewriter.replaceOpWithNewOp<IntOp>(op, type, adaptor.getOperands(),
                                         op->getAttrs());
    else
      rewriter.replaceOpWithNewOp<FloatOp>(op, type, adaptor.getOperands(),
                                           op->getAttrs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPNeg
//===----------------------------------------------------------------------===//

/// Convert an integer pop.neg(x) -> x * -1
/// and float pop.neg(x) -> llvm.fneg(x)
struct ConvertPOPNeg : public mlir::OpConversionPattern<NegOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NegOp op, NegOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DType dtype = op.getType().cast<DataTypeInterface>().resolveDType();
    if (dtype.isInt()) {
      Type type = getTypeConverter()->convertType(op.getType());
      auto zero = rewriter.create<LLVM::ConstantOp>(op.getLoc(), type, 0);
      rewriter.replaceOpWithNewOp<LLVM::SubOp>(op, zero, adaptor.getOperand());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::FNegOp>(op, adaptor.getOperand());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPAbs
//===----------------------------------------------------------------------===//

/// Convert integer pop.abs x -> llvm.abs
struct ConvertPOPAbs : public mlir::OpConversionPattern<AbsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AbsOp op, AbsOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DType dtype = op.getType().cast<DataTypeInterface>().resolveDType();
    if (dtype.isInt()) {
      Type type = adaptor.getOperand().getType();
      auto zero = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getBoolAttr(false));
      rewriter.replaceOpWithNewOp<LLVM::AbsOp>(op, type, adaptor.getOperand(),
                                               zero);
    } else {
      rewriter.replaceOpWithNewOp<LLVM::FAbsOp>(op, adaptor.getOperand());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPFMA
//===----------------------------------------------------------------------===//

/// Convert integer pop.fma(x, y, z) -> x * y + z
/// and float pop.fma(x, y, a) -> llvm.intr.fma(x, y, z)
struct ConvertPOPFMA : public mlir::OpConversionPattern<FMAOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FMAOp op, FMAOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DType dtype = op.getType().cast<DataTypeInterface>().resolveDType();
    if (dtype.isInt()) {
      auto lhs = rewriter.create<LLVM::MulOp>(op.getLoc(), adaptor.getA(),
                                              adaptor.getB());
      rewriter.replaceOpWithNewOp<LLVM::AddOp>(op, lhs, adaptor.getC());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::FMAOp>(op, adaptor.getOperands());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Trivial Conversions
//===----------------------------------------------------------------------===//

using ConvertPOPConstant = OneToOneConversion<ConstantOp, LLVM::ConstantOp>;
using ConvertPOPCopySign = OneToOneConversion<CopySignOp, LLVM::CopySignOp>;
using ConvertPOPAdd =
    OneToOneIntOrFloatConversion<AddOp, LLVM::AddOp, LLVM::FAddOp>;
using ConvertPOPSub =
    OneToOneIntOrFloatConversion<SubOp, LLVM::SubOp, LLVM::FSubOp>;
using ConvertPOPMul =
    OneToOneIntOrFloatConversion<MulOp, LLVM::MulOp, LLVM::FMulOp>;
using ConvertPOPSelect = OneToOneConversion<SelectOp, LLVM::SelectOp>;

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populatePOPToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                      mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertPOPAbs, ConvertPOPAdd, ConvertPOPConstant,
                  ConvertPOPCopySign, ConvertPOPFMA, ConvertPOPMul,
                  ConvertPOPNeg, ConvertPOPSelect, ConvertPOPSub>(
      typeConverter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertPOPToLLVMPass
    : public ConvertPOPToLLVMBase<ConvertPOPToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertPOPToLLVMPass::runOnOperation() {
  Operation *kernel = getOperation();

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<POPDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  if (indexBitwidth != mlir::kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);
  MetaToLLVMTypeConverter typeConverter(kernel->getLoc(), options);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populatePOPToLLVMPatterns(typeConverter, patterns);

  if (failed(mlir::applyPartialConversion(kernel, target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> M::KGEN::createConvertPOPToLLVMPass() {
  return std::make_unique<ConvertPOPToLLVMPass>();
}
