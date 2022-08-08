//===- ConvertKGENToLLVM.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

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
namespace LLVM = mlir::LLVM;

namespace {
class KGENToLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  KGENToLLVMTypeConverter(Location loc);

  /// Report an error or conversion failure.
  /// TODO: TypeConverter needs an error reporting mechanism.
  mlir::InFlightDiagnostic emitError(StringRef msg) {
    return mlir::emitError(loc) << msg;
  }

private:
  /// A location used to report conversion failures.
  mlir::Location loc;
};
} // end anonymous namespace

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
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, rewriter.getI64Type(), adaptor.getValue(),
        rewriter.getI64ArrayAttr(0));
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
    auto type = adaptor.getValue().getType().cast<LLVM::LLVMStructType>();
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, type.getBody()[1], adaptor.getValue(), rewriter.getI64ArrayAttr(1));
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
// KGENToLLVMTypeConverter Implementation
//===----------------------------------------------------------------------===//

static Optional<Type> getMLIRTypeForDType(MLIRContext *ctx, DType dtype) {
  // This intentionally discards signed-ness because LLVM is signless.
  if (dtype.isInt() || dtype.isSInt() || dtype.isUInt())
    return IntegerType::get(ctx, dtype.getIntegerWidthInBits());

  if (dtype.isFloat()) {
    switch (dtype.getValue()) {
    default:
      break;
    case DType::f16:
      return FloatType::getF16(ctx);
    case DType::bf16:
      return FloatType::getBF16(ctx);
    case DType::f32:
      return FloatType::getF32(ctx);
    case DType::f64:
      return FloatType::getF64(ctx);
    }
  }

  return {};
}

KGENToLLVMTypeConverter::KGENToLLVMTypeConverter(mlir::Location loc)
    : LLVMTypeConverter(loc.getContext()), loc(loc) {
  addConversion([&](Type t) -> Optional<Type> {
    emitError("could not convert ") << t << " to be an llvm-compatible type";
    return llvm::None;
  });

  // Convert a DType expression to an MLIR type.
  auto convertDType = [&](auto type) -> Optional<Type> {
    auto dtypeConst = type.getDtype().template dyn_cast<DTypeConstantAttr>();
    if (!dtypeConst) {
      emitError("dtype not fully specified: ") << type;
      return {};
    }
    return getMLIRTypeForDType(type.getContext(), dtypeConst.getDType());
  };

  // Convert a size expression to a C++ unsigned integer.
  auto convertSize = [&](auto type) -> Optional<unsigned> {
    auto size = type.getSize().template dyn_cast<IntegerAttr>();
    if (!size) {
      emitError("size not fully specified: ") << type;
      return {};
    }
    const APInt &value = size.getValue();
    assert(APInt(value.getBitWidth(), value.getLimitedValue()) == value &&
           "couldn't narrow vector size");
    return value.getLimitedValue();
  };

  // Convert scalar types directly to the dtype.
  addConversion([&](ScalarType scalar) { return convertDType(scalar); });

  // Convert pointer types to bare pointers of the dtype.
  addConversion([&](PointerType pointer) -> Optional<Type> {
    if (Optional<Type> dtype = convertDType(pointer))
      return LLVM::LLVMPointerType::get(*dtype);
    return {};
  });

  // Convert SIMD types to vector types.
  addConversion([&](SIMDType simd) -> Optional<Type> {
    Optional<Type> dtype = convertDType(simd);
    auto size = convertSize(simd);
    if (!dtype || !size)
      return {};
    return mlir::VectorType::get(*size, *dtype);
  });

  // Convert buffers to struct<(i64, ptr<T>)>.
  // TODO: Should fixed-size buffers be converted to arrays?
  addConversion([&](BufferType buffer) -> Optional<Type> {
    Optional<Type> dtype = convertDType(buffer);
    if (!dtype)
      return {};
    return LLVM::LLVMStructType::getLiteral(
        buffer.getContext(), {Builder(buffer.getContext()).getI64Type(),
                              LLVM::LLVMPointerType::get(*dtype)});
  });

  // Need basic forwarding conversions too. These are basically copied from
  // mlir/lib/Conversion/LLVMCommon/TypeConverter.cpp
  addConversion([](mlir::IntegerType integer) {
    return IntegerType::get(integer.getContext(), integer.getWidth());
  });
  addConversion([](mlir::FloatType fty) { return fty; });
}

static void populateKGENToLLVMPatterns(KGENToLLVMTypeConverter &typeConverter,
                                       mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertKGENCall, ConvertKGENKernel, ConvertKGENParamValue,
                  ConvertKGENReturn, ConvertMetaBufferAddress,
                  ConvertMetaBufferCast, ConvertMetaBufferSize,
                  ConvertMetaCastFromBuiltin, ConvertMetaCastToBuiltin,
                  ConvertUnrealizedConversionCast>(typeConverter);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
}

namespace {
#define GEN_PASS_CLASSES
#include "KGEN/KGENPasses.h.inc"
class ConvertKGENToLLVMPass
    : public ConvertKGENToLLVMBase<ConvertKGENToLLVMPass> {
public:
  void runOnOperation() override;
};
} // namespace

void ConvertKGENToLLVMPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
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
