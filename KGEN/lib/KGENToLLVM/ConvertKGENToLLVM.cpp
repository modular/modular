//===- ConvertKGENToLLVM.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENToLLVM/ConvertKGENToLLVM.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MetaDialect/MetaOps.h"
#include "KGEN/MetaDialect/MetaTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ConvertKGENKernel
//===----------------------------------------------------------------------===//

namespace {
class ConvertKGENKernel : public mlir::OpConversionPattern<KernelOp> {
public:
  using mlir::OpConversionPattern<KernelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(KernelOp kernel, KernelOpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The kernel must be fully specified for this to work, so if the kernel
    // isn't then this conversion fails.
    if (!kernel.getParamDecls().empty())
      return mlir::emitError(kernel->getLoc())
             << "cannot lower a kernel that is not fully specified.";

    // Convert the function type.
    SmallVector<Type> newArgTypes, newResultTypes;
    for (auto arg : kernel.getArgumentTypes())
      newArgTypes.push_back(getTypeConverter()->convertType(arg));
    for (auto res : kernel.getResultTypes())
      newResultTypes.push_back(getTypeConverter()->convertType(res));

    auto checkTypes = [&](ArrayRef<Type> types) -> LogicalResult {
      if (auto found = llvm::find(types, Type{}); found != types.end()) {
        size_t which = std::distance(types.begin(), found);
        return mlir::emitError(kernel.getLoc())
               << "could not convert this type: "
               << kernel.getArgument(which).getType()
               << " to be llvm-compatible.";
      }
      return success();
    };
    if (failed(checkTypes(newArgTypes)) || failed(checkTypes(newResultTypes)))
      return failure();

    Type llvmResultType = nullptr;
    if (newResultTypes.empty())
      llvmResultType = rewriter.getType<mlir::LLVM::LLVMVoidType>();
    else if (newResultTypes.size() == 1)
      llvmResultType = newResultTypes.front();
    else
      llvmResultType =
          mlir::LLVM::LLVMStructType::getLiteral(getContext(), newResultTypes);

    Type llvmType =
        mlir::LLVM::LLVMFunctionType::get(llvmResultType, newArgTypes);

    // Create the new LLVM function.
    auto llvmFunc = rewriter.create<mlir::LLVM::LLVMFuncOp>(
        kernel.getLoc(), kernel.getName(), llvmType);
    // And move the kernel's body into the new function.
    rewriter.inlineRegionBefore(kernel.getBodyRegion(0), llvmFunc.getBody(),
                                llvmFunc.end());
    if (failed(rewriter.convertRegionTypes(&llvmFunc.getBody(),
                                           *getTypeConverter())))
      return failure();

    rewriter.eraseOp(kernel);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertKGENReturn
//===----------------------------------------------------------------------===//

namespace {
class ConvertKGENReturn : public mlir::OpConversionPattern<ReturnOp> {
public:
  using mlir::OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (opAdaptor.operands().empty()) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op, mlir::ValueRange{});
      return success();
    }

    if (opAdaptor.operands().size() == 1) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op,
                                                        opAdaptor.operands());
      return success();
    }

    SmallVector<Type> resultTypes;
    resultTypes.reserve(opAdaptor.operands().size());
    for (auto t : opAdaptor.getOperands().getTypes())
      resultTypes.push_back(t);

    // Create an undef op for packing the outputs into.
    auto undefOp = rewriter.create<mlir::LLVM::UndefOp>(
        op.getLoc(),
        mlir::LLVM::LLVMStructType::getLiteral(getContext(), resultTypes));

    // Pack each operand.
    Value outputStruct = undefOp;
    for (auto operand : llvm::enumerate(opAdaptor.getOperands()))
      outputStruct = rewriter.create<mlir::LLVM::InsertValueOp>(
          op.getLoc(), outputStruct, operand.value(),
          rewriter.getIndexArrayAttr({(int64_t)operand.index()}));

    // And return the result.
    rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op, outputStruct);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertKGENParamValue
//===----------------------------------------------------------------------===//

namespace {
class ConvertKGENParamValue : public mlir::OpConversionPattern<ParamValueOp> {
public:
  using mlir::OpConversionPattern<ParamValueOp>::OpConversionPattern;

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
    : public mlir::OpConversionPattern<MetaCastToBuiltinOp> {
public:
  using mlir::OpConversionPattern<MetaCastToBuiltinOp>::OpConversionPattern;

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
    : public mlir::OpConversionPattern<MetaCastFromBuiltinOp> {
public:
  using mlir::OpConversionPattern<MetaCastFromBuiltinOp>::OpConversionPattern;

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

KGENToLLVMTypeConverter::KGENToLLVMTypeConverter() {
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

void M::populateKGENToLLVMPatterns(KGENToLLVMTypeConverter &typeConverter,
                                   mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertKGENKernel, ConvertKGENReturn, ConvertKGENParamValue,
                  ConvertMetaCastToBuiltin, ConvertMetaCastFromBuiltin>(
      typeConverter, patterns.getContext());
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
  KGENToLLVMTypeConverter typeConverter;

  populateKGENToLLVMPatterns(typeConverter, patterns);

  if (failed(mlir::applyFullConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> M::createConvertKGENToLLVMPass() {
  return std::make_unique<ConvertKGENToLLVMPass>();
}
