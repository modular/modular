//===- ConvertKGENToLLVM.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MetaDialect/MetaOps.h"
#include "KGEN/MetaDialect/MetaTypeConverter.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

//===----------------------------------------------------------------------===//
// ConvertKGENKernel
//===----------------------------------------------------------------------===//

namespace {
class ConvertKGENKernel : public mlir::ConvertOpToLLVMPattern<KernelOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(KernelOp kernel, KernelOpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The kernel must be fully specified for this to work, so if the kernel
    // isn't then this conversion fails.
    if (!kernel.getParamDecls().empty())
      return mlir::emitError(kernel->getLoc())
             << "cannot lower a kernel that is not fully specified.";

    auto funcOp = rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(
        kernel, kernel.getName(), kernel.getFunctionType());

    // And move the kernel's body into the new function.
    rewriter.inlineRegionBefore(kernel.getBodyRegion(0), funcOp.getBody(),
                                funcOp.end());
    if (failed(rewriter.convertRegionTypes(&funcOp.getBody(),
                                           *getTypeConverter())))
      return emitError(kernel.getLoc(),
                       "could not convert region types to be LLVM-compatible.");

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertKGENCall
//===----------------------------------------------------------------------===//

namespace {
/// Convert `kgen.call` to `func.call` and re-use the latter's conversion to
/// LLVM.
class ConvertKGENCall : public mlir::ConvertOpToLLVMPattern<CallOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CallOp op, CallOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getParamDecls().empty() || !op.getParamValues().empty())
      return mlir::emitError(op->getLoc())
             << "cannot lower a call op that is not fully specified.";
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, op.getCallee(), op.getResultTypes(), adaptor.getOperands());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertKGENReturn
//===----------------------------------------------------------------------===//

namespace {
class ConvertKGENReturn : public mlir::ConvertOpToLLVMPattern<ReturnOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getParameters().empty())
      return mlir::emitError(op->getLoc())
             << "cannot lower a return op that has parameters.";

    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                      opAdaptor.getOperands());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertKGENParamValue
//===----------------------------------------------------------------------===//

namespace {
class ConvertKGENParamValue
    : public mlir::ConvertOpToLLVMPattern<ParamValueOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ParamValueOp op, ParamValueOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Ensure that index types are converted.
    return LLVM::detail::oneToOneRewrite(
        op, LLVM::ConstantOp::getOperationName(), adaptor.getOperands(),
        *getTypeConverter(), rewriter);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertMetaCastToBuiltin
//===----------------------------------------------------------------------===//

namespace {
class ConvertMetaCastToBuiltin
    : public mlir::ConvertOpToLLVMPattern<MetaCastToBuiltinOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MetaCastToBuiltinOp op, MetaCastToBuiltinOpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, opAdaptor.getOperands());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertMetaCastFromBuiltin
//===----------------------------------------------------------------------===//

namespace {
class ConvertMetaCastFromBuiltin
    : public mlir::ConvertOpToLLVMPattern<MetaCastFromBuiltinOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MetaCastFromBuiltinOp op,
                  MetaCastFromBuiltinOpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, opAdaptor.getOperands());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertMetaBufferSize
//===----------------------------------------------------------------------===//

namespace {
/// Convert the size of a buffer to an `llvm.extractvalue`.
class ConvertMetaBufferSize
    : public mlir::ConvertOpToLLVMPattern<BufferSizeOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BufferSizeOp op, BufferSizeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getValue(),
                                                      0);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertMetaBufferAddress
//===----------------------------------------------------------------------===//

namespace {
/// The address of a dynamic buffer is the starting pointer. The address of a
/// fixed-size buffer is the address of the first element.
class ConvertMetaBufferAddress
    : public mlir::ConvertOpToLLVMPattern<BufferAddressOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BufferAddressOp op, BufferAddressOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getValue(),
                                                      1);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertMetaBufferCast
//===----------------------------------------------------------------------===//

namespace {
class ConvertMetaBufferCast
    : public mlir::ConvertOpToLLVMPattern<BufferCastOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BufferCastOp op, BufferCastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getBuffer());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertUnrealizedConversionCast
//===----------------------------------------------------------------------===//

/// TODO: This shouldn't be needed and should be covered by something like
/// `meta.cast_to/from_builtin`, but "builtin" now includes LLVM.
namespace {
class ConvertUnrealizedConversionCast
    : public mlir::ConvertOpToLLVMPattern<mlir::UnrealizedConversionCastOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::UnrealizedConversionCastOp op,
                  mlir::UnrealizedConversionCastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getInputs());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateKGENToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                       mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertKGENCall, ConvertKGENKernel, ConvertKGENParamValue,
                  ConvertKGENReturn, ConvertMetaBufferAddress,
                  ConvertMetaBufferCast, ConvertMetaBufferSize,
                  ConvertMetaCastFromBuiltin, ConvertMetaCastToBuiltin,
                  ConvertUnrealizedConversionCast>(typeConverter);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertKGENToLLVMPass
    : public ConvertKGENToLLVMBase<ConvertKGENToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertKGENToLLVMPass::runOnOperation() {
  ModuleOp theModule = getOperation();

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<KGENDialect, MetaDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  if (indexBitwidth != mlir::kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);
  MetaToLLVMTypeConverter typeConverter(theModule->getLoc(), options);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populateKGENToLLVMPatterns(typeConverter, patterns);

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> M::KGEN::createConvertKGENToLLVMPass() {
  return std::make_unique<ConvertKGENToLLVMPass>();
}
