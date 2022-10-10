//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ZAPDialect/ZAPDialect.h"
#include "KGEN/ZAPDialect/ZAPOps.h"
#include "KGEN/ZAPDialect/ZAPTypes.h"
#include "Support/IndexDialect/IndexOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace M;
using namespace KGEN;
using namespace POP;
using namespace ZAP;

/// The position of the buffer address in its struct representation.
static constexpr int kBufferAddressPosition = 0;
/// The position of the buffer size in its struct representation.
static constexpr int kBufferSizePosition = 1;
/// The position of the buffer dtype in its struct representation.
static constexpr int kBufferDTypePosition = 2;

namespace {

//===----------------------------------------------------------------------===//
// ConvertZAPBufferConstruct
//===----------------------------------------------------------------------===//

/// Construct a buffer struct. If either the size or dtype are dynamic, values
/// for them must be provided.
static Value constructBuffer(PatternRewriter &rewriter,
                             TypeConverter &typeConverter, Location loc,
                             BufferType type, Value ptr, Value size = {},
                             Value dtype = {}) {
  if (!size)
    size = rewriter.create<ParamConstantOp>(loc, type.getSize());
  if (!dtype)
    dtype = rewriter.create<ParamConstantOp>(loc, type.getDType());
  return rewriter.create<StructConstructOp>(
      loc, typeConverter.convertType(type), ArrayRef<Value>{ptr, size, dtype});
}

/// Convert the construction of a buffer to building the underlying struct.
struct ConvertZAPBufferConstruct
    : public mlir::OpConversionPattern<BufferConstructOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferConstructOp op, BufferConstructOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value buf = constructBuffer(rewriter, *getTypeConverter(), op.getLoc(),
                                op.getType(), adaptor.getPtr(),
                                adaptor.getSize(), adaptor.getDType());
    rewriter.replaceOp(op, buf);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferSize
//===----------------------------------------------------------------------===//

/// Extract the buffer size at element 0.
struct ConvertZAPBufferSize : public mlir::OpConversionPattern<BufferSizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferSizeOp op, BufferSizeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<GetElementOp>(op, adaptor.getBuffer(),
                                              kBufferSizePosition);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferAddress
//===----------------------------------------------------------------------===//

/// Extract the buffer pointer at element 1.
struct ConvertZAPBufferAddress
    : public mlir::OpConversionPattern<BufferAddressOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferAddressOp op, BufferAddressOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<GetElementOp>(op, adaptor.getBuffer(),
                                              kBufferAddressPosition);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferDType
//===----------------------------------------------------------------------===//

/// Extract the buffer dtype at element 2.
struct ConvertZAPBufferDType : public mlir::OpConversionPattern<BufferDTypeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferDTypeOp op, BufferDTypeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<GetElementOp>(op, adaptor.getBuffer(),
                                              kBufferDTypePosition);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferConvert
//===----------------------------------------------------------------------===//

/// When converting an buffer, we have to bitcast the pointer type. Moreover, we
/// will overwrite the size and dtype if they were respecified in the return
/// type.
struct ConvertZAPBufferConvert
    : public mlir::OpConversionPattern<BufferConvertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferConvertOp op, BufferConvertOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BufferType inputType = op.getInput().getType();
    BufferType type = op.getType();
    // If the input and output types are the same, fold away the op.
    if (inputType == type) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Bitcast the pointer if needed.
    Value ptr = rewriter.create<GetElementOp>(op.getLoc(), adaptor.getInput(),
                                              kBufferAddressPosition);
    if (type.getDType() != op.getInput().getType().getDType())
      ptr = rewriter.create<PointerBitcastOp>(
          op.getLoc(), op.getType().getPointerType(), ptr);

    // Conversion from `? -> N` means we overwrite in the output, and `N -> ?`
    // means we can use the input expression as a constant.
    Value size, dtype;
    TypedAttr sizeExpr, dtypeExpr;
    if (auto outputSize = type.getSize())
      sizeExpr = outputSize;
    else if (auto inputSize = inputType.getSize())
      sizeExpr = inputSize;
    if (auto outputDType = type.getDType())
      dtypeExpr = outputDType;
    else if (auto inputDType = inputType.getDType())
      dtypeExpr = inputDType;

    // Otherwise, if `? -> ?`, we have to query the input for the field.
    if (sizeExpr)
      size = rewriter.create<ParamConstantOp>(op.getLoc(), sizeExpr);
    else
      size = rewriter.create<GetElementOp>(op.getLoc(), adaptor.getInput(),
                                           kBufferSizePosition);
    if (dtypeExpr)
      dtype = rewriter.create<ParamConstantOp>(op.getLoc(), dtypeExpr);
    else
      dtype = rewriter.create<GetElementOp>(op.getLoc(), adaptor.getInput(),
                                            kBufferDTypePosition);

    rewriter.replaceOpWithNewOp<StructConstructOp>(
        op, getTypeConverter()->convertType(type),
        ArrayRef<Value>{ptr, size, dtype});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferStackAllocation
//===----------------------------------------------------------------------===//

struct ConvertZAPBufferStackAllocation
    : mlir::OpConversionPattern<BufferStackAllocationOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferStackAllocationOp op,
                  BufferStackAllocationOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op.getType().cast<BufferType>();
    Value ptr = rewriter.create<StackAllocationOp>(
        op.getLoc(), type.getPointerType(), type.getSize());
    Value buf = constructBuffer(rewriter, *getTypeConverter(), op.getLoc(),
                                op.getType(), ptr);
    rewriter.replaceOp(op, buf);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferConstant
//===----------------------------------------------------------------------===//

struct ConvertZAPBufferConstant : mlir::OpConversionPattern<BufferConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferConstantOp op, BufferConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BufferType type = op.getType();
    auto elType = ScalarType::get(type.getDType());
    Value global = rewriter.create<GlobalConstantOp>(
        op.getLoc(),
        PointerType::get(POP::ArrayType::get(type.getSize(), elType)),
        op.getValues());
    Value ptr = rewriter.create<PointerBitcastOp>(
        op.getLoc(), PointerType::get(elType), global);
    Value buf =
        constructBuffer(rewriter, *getTypeConverter(), op.getLoc(), type, ptr);
    rewriter.replaceOp(op, buf);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferLoad
//===----------------------------------------------------------------------===//

struct ConvertZAPBufferLoad : mlir::OpConversionPattern<BufferLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferLoadOp op, BufferLoadOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value base = rewriter.create<GetElementOp>(op.getLoc(), adaptor.getBuffer(),
                                               kBufferAddressPosition);
    Value ptr =
        rewriter.create<OffsetOp>(op.getLoc(), base, adaptor.getPosition());
    rewriter.replaceOpWithNewOp<LoadOp>(op, ptr, /*alignment=*/None);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferStore
//===----------------------------------------------------------------------===//

struct ConvertZAPBufferStore : mlir::OpConversionPattern<BufferStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferStoreOp op, BufferStoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value base = rewriter.create<GetElementOp>(op.getLoc(), adaptor.getBuffer(),
                                               kBufferAddressPosition);
    Value ptr =
        rewriter.create<OffsetOp>(op.getLoc(), base, adaptor.getPosition());
    rewriter.replaceOpWithNewOp<StoreOp>(op, adaptor.getValue(), ptr,
                                         /*alignment=*/None);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferSIMDLoad
//===----------------------------------------------------------------------===//

struct ConvertZAPBufferSIMDLoad : mlir::OpConversionPattern<BufferSIMDLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferSIMDLoadOp op, BufferSIMDLoadOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value base = rewriter.create<GetElementOp>(op.getLoc(), adaptor.getBuffer(),
                                               kBufferAddressPosition);
    Value ptr =
        rewriter.create<OffsetOp>(op.getLoc(), base, adaptor.getPosition());
    Value bitcastPtr = rewriter.create<PointerBitcastOp>(
        op.getLoc(), PointerType::get(TypeConstantAttr::get(op.getType())),
        ptr);
    // We set the alignment to 1 to force LLVM to generate unaligned loads.
    rewriter.replaceOpWithNewOp<LoadOp>(op, bitcastPtr, /*alignment=*/1);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferSIMDStore
//===----------------------------------------------------------------------===//

struct ConvertZAPBufferSIMDStore
    : mlir::OpConversionPattern<BufferSIMDStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferSIMDStoreOp op, BufferSIMDStoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value base = rewriter.create<GetElementOp>(op.getLoc(), adaptor.getBuffer(),
                                               kBufferAddressPosition);
    Value ptr =
        rewriter.create<OffsetOp>(op.getLoc(), base, adaptor.getPosition());
    Value bitcastPtr = rewriter.create<PointerBitcastOp>(
        op.getLoc(), PointerType::get(op.getValue().getType()), ptr);
    // We set the alignment to 1 to force LLVM to generate unaligned stores.
    rewriter.replaceOpWithNewOp<StoreOp>(op, adaptor.getValue(), bitcastPtr,
                                         /*alignment=*/1);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPPrint
//===----------------------------------------------------------------------===//

struct ConvertZAPPrint : mlir::OpConversionPattern<PrintOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintOp op, PrintOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Lower the string into the a global constant. Null-terminate the string.
    SmallVector<char> fmtStr;
    fmtStr.reserve(op.getFmt().size() + 1);
    llvm::append_range(fmtStr, op.getFmt());
    fmtStr.push_back('\0');
    auto values = IntArrayElementsAttr::get(
        op.getContext(), ArrayRef<char>(fmtStr), IntegerType::Signed);
    auto charType = rewriter.getType<ScalarType>(DType::si8);
    Value fmtData = rewriter.create<GlobalConstantOp>(
        op.getLoc(),
        PointerType::get(POP::ArrayType::get(fmtStr.size(), charType)), values);
    Value fmt = rewriter.create<PointerBitcastOp>(
        op.getLoc(), PointerType::get(charType), fmtData);

    // Create the invocation to `printf`.
    SmallVector<Value> operands;
    operands.reserve(op.getNumOperands() + 1);
    operands.push_back(fmt);
    llvm::append_range(operands, adaptor.getOperands());
    rewriter.replaceOpWithNewOp<ExternalCallOp>(
        op, TypeRange(), "printf", operands,
        TypeAttr::get(rewriter.getFunctionType(fmt.getType(), {})));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Signature Conversion
//===----------------------------------------------------------------------===//

struct ConvertCallSignature : public mlir::OpConversionPattern<CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CallOp op, CallOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    resultTypes.reserve(op.getNumResults());
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();
    auto call = rewriter.create<CallOp>(
        op.getLoc(), resultTypes, op.getCalleeAttr(), op.getParamValuesAttr(),
        op.getParamDeclsAttr(), adaptor.getOperands(), op->getNumRegions());
    // Move all the regions over.
    for (auto [prev, region] : llvm::zip(op.getRegions(), call.getRegions()))
      region->takeBody(*prev);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

template <typename OpT>
struct ConvertInterfaceSignature : public mlir::OpConversionPattern<OpT> {
  using mlir::OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType type = op.getFunctionType();
    TypeConverter::SignatureConversion inputs(type.getNumInputs()),
        results(type.getNumResults());
    if (failed(this->getTypeConverter()->convertSignatureArgs(type.getInputs(),
                                                              inputs)) ||
        failed(this->getTypeConverter()->convertSignatureArgs(type.getResults(),
                                                              results)))
      return failure();
    rewriter.updateRootInPlace(op, [&] {
      op.setType(rewriter.getFunctionType(inputs.getConvertedTypes(),
                                          results.getConvertedTypes()));
    });
    return success();
  }
};

template <typename Op>
struct ConvertSignature : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Block *> result = rewriter.convertRegionTypes(
        &op.getBodyRegion(), *this->getTypeConverter());
    if (failed(result))
      return failure();

    TypeConverter::SignatureConversion results(op.getNumResults());
    if (failed(this->getTypeConverter()->convertSignatureArgs(
            op.getResultTypes(), results)))
      return failure();

    rewriter.updateRootInPlace(op, [&] {
      op.setType(rewriter.getFunctionType(result.value()->getArgumentTypes(),
                                          results.getConvertedTypes()));
    });
    return success();
  }
};

struct ConvertResultSignature : public mlir::OpConversionPattern<ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct ConvertRebind : public mlir::OpConversionPattern<RebindOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RebindOp op, RebindOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<RebindOp>(
        op, getTypeConverter()->convertType(op.getType()), adaptor.getInput());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateZAPToPOPPatterns(TypeConverter &converter,
                                     RewritePatternSet &patterns) {
  patterns.insert<
      // clang-format off

      // Signature type conversions.
      ConvertCallSignature,
      ConvertInterfaceSignature<GeneratorInterfaceOp>,
      ConvertInterfaceSignature<PrecompiledLLVMOp>,
      ConvertInterfaceSignature<PrecompiledObjectOp>,
      ConvertSignature<GeneratorOp>,
      ConvertSignature<FuncOp>,
      ConvertResultSignature,
      ConvertRebind,

      // Op conversions.
      ConvertZAPBufferAddress,
      ConvertZAPBufferConstant,
      ConvertZAPBufferConstruct,
      ConvertZAPBufferConvert,
      ConvertZAPBufferDType,
      ConvertZAPBufferLoad,
      ConvertZAPBufferSIMDLoad,
      ConvertZAPBufferSIMDStore,
      ConvertZAPBufferSize,
      ConvertZAPBufferStackAllocation,
      ConvertZAPBufferStore,
      ConvertZAPPrint

      // clang-format on
      >(converter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERZAPTOPOP
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerZAPToPOPPass
    : public KGEN::impl::LowerZAPToPOPBase<LowerZAPToPOPPass> {
  using LowerZAPToPOPBase::LowerZAPToPOPBase;

  void runOnOperation() override;
};
} // namespace

void LowerZAPToPOPPass::runOnOperation() {
  ModuleOp theModule = getOperation();

  // Configure the type converter.
  TypeConverter typeConverter;
  typeConverter.addConversion([=](Type type) -> Optional<Type> {
    auto buf = dyn_cast<BufferType>(type);
    if (!buf)
      return type;
    // Convert buffer types to a struct of (pointer, index, dtype).
    return StructType::get({buf.getPointerType(),
                            IndexType::get(buf.getContext()),
                            DTypeType::get(buf.getContext())});
  });

  // Configure dialect conversion
  ConversionTarget target(getContext());
  target.addIllegalDialect<ZAPDialect>();
  target.addLegalDialect<POPDialect, KGENDialect>();

  auto isZAPType = [&](Type type) { return type.isa<BufferType>(); };

  // Dynamically legalize KGEN operations that can interact with any parametric
  // type, including ZAP types.
  target.addDynamicallyLegalOp<GeneratorInterfaceOp, GeneratorOp, FuncOp,
                               PrecompiledLLVMOp, PrecompiledObjectOp>(
      [&](Operation *op) {
        FunctionType type = cast<KGENDeclInterface>(op).getFunctionType();
        return llvm::none_of(type.getInputs(), isZAPType) &&
               llvm::none_of(type.getResults(), isZAPType);
      });
  target.addDynamicallyLegalOp<CallOp, RebindOp, ReturnOp>([&](Operation *op) {
    return llvm::none_of(op->getOperandTypes(), isZAPType) &&
           llvm::none_of(op->getResultTypes(), isZAPType);
  });

  // Populate patterns
  RewritePatternSet patterns(&getContext());
  populateZAPToPOPPatterns(typeConverter, patterns);

  // Run the conversion.
  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();
}
