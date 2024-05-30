//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_GENERICOPCONVERSION_H
#define SUPPORT_COMPILER_GENERICOPCONVERSION_H

#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Transforms/DialectConversion.h"

namespace M {

/// Converts a graph-like op of type `SourceOp` to a graph-like op of type
/// `TargetOp` in another dialect.
template <typename SourceOp, typename TargetOp>
ErrorOr<TargetOp> convertGraphOp(SourceOp op,
                                 ConversionPatternRewriter &rewriter,
                                 const TypeConverter *typeConverter) {
  auto fnType = op.getFunctionType();

  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  for (const auto &argType : enumerate(fnType.getInputs())) {
    auto convertedType = typeConverter->convertType(argType.value());
    if (!convertedType)
      return Error("argument type cannot be converted");

    signatureConverter.addInputs(argType.index(), convertedType);
  }

  SmallVector<Type> resultTypes;
  if (failed(typeConverter->convertTypes(fnType.getResults(), resultTypes)))
    return Error("result types cannot be converted");

  if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter,
                                         &signatureConverter)))
    return Error("block argument types cannot be converted");

  auto newOp = rewriter.create<TargetOp>(op.getLoc(), op.getName(),
                                         signatureConverter.getConvertedTypes(),
                                         resultTypes);

  // The builder should automatically add a body, so we remove it, and replace
  // it with the body of the source op.
  newOp.getBody().front().erase();
  rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                              newOp.getBody().end());

  rewriter.eraseOp(op);
  return newOp;
}

/// Conversion pattern to use when converting containers for graph-like models.
template <typename SourceOp, typename TargetOp>
class GenericGraphOpConversion final : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOpOr = convertGraphOp<SourceOp, TargetOp>(op, rewriter,
                                                      this->getTypeConverter());
    if (newOpOr.isError())
      return rewriter.notifyMatchFailure(op, newOpOr.getError());

    return success();
  }
};

/// Conversion pattern to use when converting terminators of graph-like
/// containers.
template <typename SourceOp, typename TargetOp>
class GenericOutputOpConversion final : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getOperands());
    return success();
  }
};

} // namespace M

#endif // SUPPORT_COMPILER_GENERICOPCONVERSION_H
