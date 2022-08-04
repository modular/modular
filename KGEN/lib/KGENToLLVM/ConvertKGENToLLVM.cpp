//===- ConvertKGENToLLVM.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENToLLVM/ConvertKGENToLLVM.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MetaDialect/MetaOps.h"
#include "KGEN/MetaDialect/MetaTypes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

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

namespace {
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
  using mlir::ConvertOpToLLVMPattern<ReturnOp>::ConvertOpToLLVMPattern;

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
  using mlir::ConvertOpToLLVMPattern<ParamValueOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ParamValueOp op, ParamValueOpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(op, op.getType(),
                                                        op.getValue());
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
  using mlir::ConvertOpToLLVMPattern<
      MetaCastToBuiltinOp>::ConvertOpToLLVMPattern;

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
  using mlir::ConvertOpToLLVMPattern<
      MetaCastFromBuiltinOp>::ConvertOpToLLVMPattern;

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
// KGENToLLVMTypeConverter Implementation
//===----------------------------------------------------------------------===//

static mlir::Type getMLIRTypeForDType(MLIRContext *ctx, DType dtype) {
  // This intentionally discards signed-ness because LLVM is signless.
  if (dtype.isInt() || dtype.isSInt() || dtype.isUInt())
    return mlir::IntegerType::get(ctx, dtype.getIntegerWidthInBits());

  if (dtype.isFloat()) {
    switch (dtype.getValue()) {
    default:
      break;
    case DType::f16:
      return mlir::FloatType::getF16(ctx);
    case DType::bf16:
      return mlir::FloatType::getBF16(ctx);
    case DType::f32:
      return mlir::FloatType::getF32(ctx);
    case DType::f64:
      return mlir::FloatType::getF64(ctx);
    }
  }

  return {};
}

KGENToLLVMTypeConverter::KGENToLLVMTypeConverter(mlir::Location loc)
    : LLVMTypeConverter(loc.getContext()), loc(loc) {
  addConversion([&](mlir::Type t) -> Optional<mlir::Type> {
    emitError("could not convert ") << t << " to be an llvm-compatible type";
    return llvm::None;
  });
  addConversion([](ScalarType scalar) -> Optional<Type> {
    DType dtype = scalar.getDtype().cast<DTypeConstantAttr>().getDType();
    auto outType = getMLIRTypeForDType(scalar.getContext(), dtype);
    if (!outType)
      return llvm::None;
    return outType;
  });
  // Need basic forwarding conversions too. These are basically copied from
  // mlir/lib/Conversion/LLVMCommon/TypeConverter.cpp
  addConversion([](mlir::IntegerType integer) {
    return IntegerType::get(integer.getContext(), integer.getWidth());
  });
  addConversion([](mlir::FloatType fty) { return fty; });
}

void M::KGEN::populateKGENToLLVMPatterns(KGENToLLVMTypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertKGENKernel, ConvertKGENCall, ConvertKGENReturn,
                  ConvertKGENParamValue, ConvertMetaCastToBuiltin,
                  ConvertMetaCastFromBuiltin>(typeConverter);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
}

namespace {
#define GEN_PASS_CLASSES
#include "KGEN/KGENToLLVM/ConvertKGENToLLVM.h.inc"
class ConvertKGENToLLVMPass
    : public ConvertKGENToLLVMBase<ConvertKGENToLLVMPass> {
public:
  void runOnOperation() override;
};
} // namespace

void ConvertKGENToLLVMPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  target.addIllegalDialect<KGENDialect, MetaDialect>();

  mlir::RewritePatternSet patterns(&getContext());
  KGENToLLVMTypeConverter typeConverter(theModule->getLoc());

  populateKGENToLLVMPatterns(typeConverter, patterns);

  if (failed(mlir::applyFullConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> M::KGEN::createConvertKGENToLLVMPass() {
  return std::make_unique<ConvertKGENToLLVMPass>();
}
