//===- ConvertKGENToLLVM.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MetaDialect/MetaOps.h"
#include "KGEN/MetaDialect/MetaTypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

namespace {

//===----------------------------------------------------------------------===//
// ConvertKGENKernel
//===----------------------------------------------------------------------===//

class ConvertKGENKernel : public mlir::ConvertOpToLLVMPattern<KernelOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(KernelOp kernel, KernelOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the kernel signature.
    TypeConverter::SignatureConversion result(kernel.getNumArguments());
    Type funcType = getTypeConverter()->convertFunctionSignature(
        kernel.getFunctionType(),
        /*isVariadic=*/false, result);
    if (!funcType)
      return emitError(kernel.getLoc(), "failed to convert kernel signature");

    // Create the LLVM function.
    auto funcOp = rewriter.replaceOpWithNewOp<LLVM::LLVMFuncOp>(
        kernel, kernel.getNameAttr(), funcType);

    // And move the kernel's body into the new function.
    rewriter.inlineRegionBefore(kernel.getBodyRegion(0), funcOp.getBody(),
                                funcOp.end());
    (void)rewriter.convertRegionTypes(&funcOp.getBody(), *getTypeConverter());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENCall
//===----------------------------------------------------------------------===//

/// Convert `kgen.call` to `func.call` and re-use the latter's conversion to
/// LLVM.
class ConvertKGENCall : public mlir::ConvertOpToLLVMPattern<CallOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CallOp op, CallOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the result types.
    SmallVector<Type> types = llvm::to_vector(op.getResultTypes());
    if (!types.empty()) {
      types.assign({getTypeConverter()->packFunctionResults(types)});
      if (!types.back())
        return emitError(op.getLoc(), "failed to convert call result type");
    }

    // Create the LLVM call operation.
    auto llvmCall = rewriter.create<LLVM::CallOp>(
        op.getLoc(), types, op.getCalleeAttr(), adaptor.getOperands());

    // Unpack the struct if necessary.
    SmallVector<Value> results;
    if (op.getNumResults() <= 1) {
      llvm::append_range(results, llvmCall.getResults());
    } else {
      results.reserve(op.getNumResults());
      for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
        results.push_back(rewriter.create<LLVM::ExtractValueOp>(
            op.getLoc(), llvmCall.getResult(), i));
      }
    }

    // Replace the call operation.
    rewriter.replaceOp(op, results);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENReturn
//===----------------------------------------------------------------------===//

class ConvertKGENReturn : public mlir::ConvertOpToLLVMPattern<ReturnOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If the results don't need to be packed, create the LLVM return.
    if (op.getNumOperands() <= 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(),
                                                  adaptor.getOperands());
      return success();
    }

    // Pack the function results in a struct.
    Type type = getTypeConverter()->packFunctionResults(op.getOperandTypes());
    if (!type)
      return emitError(op.getLoc(), "failed to convert return types");
    Value result = rewriter.create<LLVM::UndefOp>(op.getLoc(), type);
    for (auto &it : llvm::enumerate(adaptor.getOperands())) {
      result = rewriter.create<LLVM::InsertValueOp>(op.getLoc(), result,
                                                    it.value(), it.index());
    }

    // Create the LLVM return.
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENParamValue
//===----------------------------------------------------------------------===//

class ConvertKGENParamConstant
    : public mlir::ConvertOpToLLVMPattern<ParamConstantOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ParamConstantOp op, ParamConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto dtype = op.getValue().dyn_cast<DTypeConstantAttr>()) {
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, rewriter.getI8Type(), dtype.getDType().getValue());
    } else if (auto attr = op.getValue().dyn_cast<TypedAttr>()) {
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, getTypeConverter()->convertType(attr.getType()), attr);
    } else {
      return rewriter.notifyMatchFailure(op, "unknown parameter value type");
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMetaCastToBuiltin
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// ConvertMetaCastFromBuiltin
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// ConvertMetaBufferSize
//===----------------------------------------------------------------------===//

/// Convert the size of a buffer with a known size to a constant. Otherwise,
/// generate an `llvm.extractvalue`.
class ConvertMetaBufferSize
    : public mlir::ConvertOpToLLVMPattern<BufferSizeOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BufferSizeOp op, BufferSizeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BufferDescriptor buffer(op.getValue().getType().cast<BufferType>());
    if (Optional<int64_t> size = buffer.getSize()) {
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, getTypeConverter()->getIndexType(), *size);
    } else {
      rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getValue(),
                                                        *buffer.getSizeIndex());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMetaBufferDType
//===----------------------------------------------------------------------===//

/// Convert the data type of a buffer with a known data type to a constant with
/// the value of the `DType::getValue` enum. Otherwise, generate an
/// `llvm.extractvalue`.
class ConvertMetaBufferDType
    : public mlir::ConvertOpToLLVMPattern<BufferDTypeOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BufferDTypeOp op, BufferDTypeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BufferDescriptor buffer(op.getValue().getType().cast<BufferType>());
    if (Optional<DType> dtype = buffer.getDType()) {
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, rewriter.getI8IntegerAttr(dtype->getValue()));
    } else {
      rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
          op, adaptor.getValue(), *buffer.getDTypeIndex());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMetaBufferAddress
//===----------------------------------------------------------------------===//

/// Convert the address of a buffer with known size and element type to itself,
/// since those buffers are converted to raw pointers. Otherwise, generate an
/// `llvm.extractvalue` of the pointer field.
class ConvertMetaBufferAddress
    : public mlir::ConvertOpToLLVMPattern<BufferAddressOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BufferAddressOp op, BufferAddressOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BufferDescriptor buffer(op.getValue().getType().cast<BufferType>());
    if (buffer.isBarePtr()) {
      rewriter.replaceOp(op, adaptor.getValue());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getValue(),
                                                        *buffer.getPtrIndex());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMetaBufferCast
//===----------------------------------------------------------------------===//

/// A buffer cast can cast between a buffer with an unspecified size or element
/// type to one with specified size or element type. When that happens, generate
/// the necessary struct unpacking and repacking and bitcasts.
class ConvertMetaBufferCast
    : public mlir::ConvertOpToLLVMPattern<BufferCastOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BufferCastOp op, BufferCastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto in = op.getBuffer().getType().cast<BufferType>();
    auto out = op.getResult().getType().cast<BufferType>();

    // If the input and output types are the same, fold away the op.
    if (in == out) {
      rewriter.replaceOp(op, adaptor.getBuffer());
      return success();
    }

    // Convert the pointer.
    Type inPtrType = getMLIRTypeForDType(op.getContext(), in.resolveDType())
                         .value_or(rewriter.getI8Type());
    Value inPtr = rewriter.create<BufferAddressOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(inPtrType), op.getBuffer());
    DType outDType = out.resolveDType();
    Type outPtrType = getMLIRTypeForDType(op.getContext(), outDType)
                          .value_or(rewriter.getI8Type());
    Value outPtr = inPtr;
    if (outPtrType != inPtrType) {
      outPtr = rewriter.create<LLVM::BitcastOp>(
          op.getLoc(), LLVM::LLVMPointerType::get(outPtrType), inPtr);
    }

    // Bare pointer output.
    BufferDescriptor buffer(out);
    if (buffer.isBarePtr()) {
      rewriter.replaceOp(op, outPtr);
      return success();
    }

    // Create the new struct.
    Value outBuffer = rewriter.create<LLVM::UndefOp>(
        op.getLoc(), getTypeConverter()->convertType(out));

    // If the output buffer has an unknown size, insert it as the first field.
    if (Optional<int64_t> index = buffer.getSizeIndex()) {
      Value inSize = rewriter.create<BufferSizeOp>(
          op.getLoc(), getTypeConverter()->getIndexType(), op.getBuffer());
      outBuffer = rewriter.create<LLVM::InsertValueOp>(op.getLoc(), outBuffer,
                                                       inSize, *index);
    }

    // If the output buffer has an unknown data type, insert it. Its position if
    // offset by 1 if the size is unknown.
    if (Optional<int64_t> index = buffer.getDTypeIndex()) {
      Value inDType = rewriter.create<BufferDTypeOp>(
          op.getLoc(), rewriter.getI8Type(), op.getBuffer());
      outBuffer = rewriter.create<LLVM::InsertValueOp>(op.getLoc(), outBuffer,
                                                       inDType, *index);
    }

    // Insert the casted pointer. Its position is offset by 1 for each unknown
    // size or dtype.
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(op, outBuffer, outPtr,
                                                     *buffer.getPtrIndex());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateKGENToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                       mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertKGENCall, ConvertKGENKernel, ConvertKGENParamConstant,
                  ConvertKGENReturn, ConvertMetaBufferAddress,
                  ConvertMetaBufferCast, ConvertMetaBufferDType,
                  ConvertMetaBufferSize, ConvertMetaCastFromBuiltin,
                  ConvertMetaCastToBuiltin>(typeConverter);
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
