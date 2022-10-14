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
#include "Support/IndexDialect/IndexDialect.h"
#include "Support/IndexDialect/IndexOps.h"
#include "Support/MDialect/MTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

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

/// The position of the tensor address in its struct representation.
static constexpr int kTensorAddressPosition = 0;
/// The position of the tensor rank in its struct representation.
static constexpr int kTensorRankPosition = 1;
/// The position of the tensor shape in its struct representation.
static constexpr int kTensorShapePosition = 2;
/// The position of the tensor dtype in its struct representation.
static constexpr int kTensorDTypePosition = 3;

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
    rewriter.replaceOpWithNewOp<StructGetOp>(op, adaptor.getBuffer(),
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
    rewriter.replaceOpWithNewOp<StructGetOp>(op, adaptor.getBuffer(),
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
    rewriter.replaceOpWithNewOp<StructGetOp>(op, adaptor.getBuffer(),
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
    Value ptr = rewriter.create<StructGetOp>(op.getLoc(), adaptor.getInput(),
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
      size = rewriter.create<StructGetOp>(op.getLoc(), adaptor.getInput(),
                                          kBufferSizePosition);
    if (dtypeExpr)
      dtype = rewriter.create<ParamConstantOp>(op.getLoc(), dtypeExpr);
    else
      dtype = rewriter.create<StructGetOp>(op.getLoc(), adaptor.getInput(),
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
    Value base = rewriter.create<StructGetOp>(op.getLoc(), adaptor.getBuffer(),
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
    Value base = rewriter.create<StructGetOp>(op.getLoc(), adaptor.getBuffer(),
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
    Value base = rewriter.create<StructGetOp>(op.getLoc(), adaptor.getBuffer(),
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
    Value base = rewriter.create<StructGetOp>(op.getLoc(), adaptor.getBuffer(),
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

/// Lower the string into the a global constant. Null-terminate the string.
static Value lowerStringToGlobalConstant(Operation *op, StringRef str,
                                         ConversionPatternRewriter &rewriter) {
  SmallVector<char> nullTerminatedStr(str.begin(), str.end());
  nullTerminatedStr.push_back('\0');
  auto values = IntArrayElementsAttr::get(rewriter.getContext(),
                                          ArrayRef<char>(nullTerminatedStr),
                                          IntegerType::Signed);
  auto charType = rewriter.getType<ScalarType>(DType::si8);
  Value nullTerminatedStrData = rewriter.create<GlobalConstantOp>(
      op->getLoc(),
      PointerType::get(POP::ArrayType::get(nullTerminatedStr.size(), charType)),
      values);
  return rewriter.create<PointerBitcastOp>(
      op->getLoc(), PointerType::get(charType), nullTerminatedStrData);
}

struct ConvertZAPPrint : mlir::OpConversionPattern<PrintOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintOp op, PrintOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Lower the format into the a global constant.
    Value fmt = lowerStringToGlobalConstant(op, op.getFmt(), rewriter);
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
// ConvertZAPAssert
//===----------------------------------------------------------------------===//

struct ConvertZAPDebugAssert : mlir::OpConversionPattern<DebugAssertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DebugAssertOp op, DebugAssertOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the function Name.
    auto functionNameStr = op->getParentOfType<mlir::FunctionOpInterface>()
                               ->getName()
                               .getStringRef();

    // Get the file/line information if available.
    std::string locationStr;
    if (auto fileLineCol = op->getLoc().dyn_cast<mlir::FileLineColLoc>()) {
      locationStr = (Twine(fileLineCol.getFilename()) + ":" +
                     Twine(fileLineCol.getLine()))
                        .str();
    } else {
      llvm::raw_string_ostream os(locationStr);
      op->getLoc().print(os);
    }

    // Convert into MLIR Values.
    Value functionName =
        lowerStringToGlobalConstant(op, functionNameStr, rewriter);
    Value filenameVal = lowerStringToGlobalConstant(op, locationStr, rewriter);
    Value message = lowerStringToGlobalConstant(op, op.getMsg(), rewriter);

    // Call into the CompilerRT assert function.
    rewriter.replaceOpWithNewOp<ExternalCallOp>(
        op, TypeRange(), "KGEN_CompilerRT_DebugAssert",
        ValueRange{op.getCond(), functionName, filenameVal, message}, nullptr);

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

//===----------------------------------------------------------------------===//
// ConvertZAPTensorConstruct
//===----------------------------------------------------------------------===//

/// Construct a buffer struct. If either the size or dtype are dynamic, values
/// for them must be provided.
static Value constructTensor(PatternRewriter &rewriter,
                             TypeConverter &typeConverter, Location loc,
                             TensorType type, Value ptr, size_t rank,
                             ValueRange shape = {}, Value dtype = {}) {
  IndexType indexType = rewriter.getIndexType();
  // Initialize the shape values with all zeros.
  auto zeroIndex = rewriter.create<index::ConstantOp>(loc, 0);
  std::array<Value, TensorType::getMaximumRank()> shapeValues;
  shapeValues.fill(zeroIndex);
  // Fill the shape values with the provided values or the constants specified
  // by the type.
  for (size_t i = 0, shapeParamOffset = 0; i < rank; ++i) {
    if (type.getShape()[i])
      shapeValues[i] = rewriter.create<index::ConstantOp>(
          loc, indexType, type.getShape()[i].cast<IntegerAttr>());
    else
      shapeValues[i] = shape[shapeParamOffset++];
  }
  // Create the shape array.
  auto shapeArray = rewriter.create<ArrayCreateOp>(loc, shapeValues);

  auto rankVal = rewriter.create<index::ConstantOp>(loc, type.getRank());
  if (!dtype)
    dtype = rewriter.create<ParamConstantOp>(loc, type.getDType());
  return rewriter.create<StructConstructOp>(
      loc, typeConverter.convertType(type),
      ValueRange{ptr, rankVal, shapeArray, dtype});
}

/// Convert the construction of a buffer to building the underlying struct.
struct ConvertZAPTensorConstruct
    : public mlir::OpConversionPattern<TensorConstructOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorConstructOp op, TensorConstructOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value tensor =
        constructTensor(rewriter, *getTypeConverter(), op.getLoc(),
                        op.getType(), adaptor.getPtr(), op.getType().getRank(),
                        adaptor.getShape(), adaptor.getDType());
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPTensorAddress
//===----------------------------------------------------------------------===//

/// Extract the tensor address.
struct ConvertZAPTensorAddress
    : public mlir::OpConversionPattern<TensorAddressOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorAddressOp op, TensorAddressOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<StructGetOp>(op, adaptor.getTensor(),
                                             kTensorAddressPosition);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPTensorRank
//===----------------------------------------------------------------------===//

/// Extract the tensor rank.
struct ConvertZAPTensorRank : public mlir::OpConversionPattern<TensorRankOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorRankOp op, TensorRankOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<StructGetOp>(op, adaptor.getTensor(),
                                             kTensorRankPosition);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPTensorDim
//===----------------------------------------------------------------------===//

/// Extract the buffer dimension at index provided.
struct ConvertZAPTensorDim : public mlir::OpConversionPattern<TensorDimOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorDimOp op, TensorDimOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value shape = rewriter.create<StructGetOp>(
        op->getLoc(), adaptor.getTensor(), kTensorShapePosition);
    rewriter.replaceOpWithNewOp<ArrayGetOp>(op, op.getType(), shape,
                                            op.getIndex());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPTensorDType
//===----------------------------------------------------------------------===//

/// Extract the tensor rank.
struct ConvertZAPTensorDType : public mlir::OpConversionPattern<TensorDTypeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorDTypeOp op, TensorDTypeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<StructGetOp>(op, adaptor.getTensor(),
                                             kTensorDTypePosition);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPTensorLoad
//===----------------------------------------------------------------------===//

/// Generates pop/index ops that computes the linearized index expression
///  given a tensor shape array and an index array, by essentially computing
///  a dot product between the index and shape vectors.
/// Assumes that the tensor is in row-major contiguous layout.
/// This function also assumes the index and shape layout as
///  [x,y,z,0,0], with x being the highest order dimension
///  i.e. outermost dimension.
static Value linearizeContiguousIndices(ConversionPatternRewriter &rewriter,
                                        Location loc, IndexType indexType,
                                        Value shapeArray,
                                        OperandRange indexArray) {

  // This function computes the dot product between the index and
  //  shape vector. Example:
  //      zap.tensor.load %tens[x, y, z] : !zap.tensor<[a,b,c], dtype>
  // will compute the `accumulatedOffset` by
  //   accumulatedOffset = z + c*(y + b*x)
  // in 2 iterations, each iteration multiplies a number from the shape
  // list and adds a number from the index list.

  // Initialize `accumulatedOffset` with the innermost index.
  //  e.g. add the `x` term in the example above.
  Value accumulatedOffset = *indexArray.begin();

  // Iterate through indices and create the multiply and add
  //   in each iteration.
  for (auto [indexPosition, indexValue] :
       llvm::drop_begin(llvm::enumerate(indexArray))) {
    // Dimension size at current position from the shape list.
    //  e.g. load the `b` term from example above.
    Value positionSize = rewriter.create<ArrayGetOp>(
        loc, indexType, shapeArray, rewriter.getIndexAttr(indexPosition));

    // Multiply by the current size, e.g. x-> b*x from example above.
    accumulatedOffset = rewriter.create<index::MulOp>(
        loc, indexType, accumulatedOffset, positionSize);

    // Add the current index, e.g. b*x -> b*x + y from example above.
    accumulatedOffset = rewriter.create<index::AddOp>(
        loc, indexType, accumulatedOffset, indexValue);
  }

  return accumulatedOffset;
}

/// Load a scalar value from tensor given a list of position indices.
struct ConvertZAPTensorLoad : public mlir::OpConversionPattern<TensorLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorLoadOp op, TensorLoadOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value shapeArray = rewriter.create<StructGetOp>(
        op->getLoc(), adaptor.getTensor(), kTensorShapePosition);

    auto offset = linearizeContiguousIndices(rewriter, op->getLoc(),
                                             rewriter.getIndexType(),
                                             shapeArray, op.getPositions());

    Value base = rewriter.create<StructGetOp>(op.getLoc(), adaptor.getTensor(),
                                              kTensorAddressPosition);

    Value ptr = rewriter.create<OffsetOp>(op.getLoc(), base, offset);

    rewriter.replaceOpWithNewOp<LoadOp>(op, ptr, /*alignment=*/None);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPTensorStore
//===----------------------------------------------------------------------===//

/// Store a scalar value into tensor given a list of position indices.
struct ConvertZAPTensorStore : public mlir::OpConversionPattern<TensorStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorStoreOp op, TensorStoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value shapeArray = rewriter.create<StructGetOp>(
        op->getLoc(), adaptor.getTensor(), kTensorShapePosition);

    // Assumes that the tensor op verifier already checked
    //  the equality of index and shape list sizes.
    auto offset = linearizeContiguousIndices(rewriter, op->getLoc(),
                                             rewriter.getIndexType(),
                                             shapeArray, op.getPositions());

    Value base = rewriter.create<StructGetOp>(op.getLoc(), adaptor.getTensor(),
                                              kTensorAddressPosition);

    Value ptr = rewriter.create<OffsetOp>(op.getLoc(), base, offset);

    rewriter.replaceOpWithNewOp<StoreOp>(op, adaptor.getValue(), ptr,
                                         /*alignment=*/None);
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
      ConvertZAPBufferSIMDLoad,
      ConvertZAPBufferSIMDStore,
      ConvertZAPBufferSIMDStore,
      ConvertZAPBufferSize,
      ConvertZAPBufferStackAllocation,
      ConvertZAPBufferStore,
      ConvertZAPDebugAssert,
      ConvertZAPPrint,
      ConvertZAPTensorAddress,
      ConvertZAPTensorConstruct,
      ConvertZAPTensorDim,
      ConvertZAPTensorDType,
      ConvertZAPTensorLoad,
      ConvertZAPTensorRank,
      ConvertZAPTensorStore

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
    return TypeSwitch<Type, Optional<Type>>(type)
        .Case([&](BufferType buf) {
          // Convert buffer types to a struct of (pointer, index, dtype).
          return StructType::get({buf.getPointerType(),
                                  IndexType::get(buf.getContext()),
                                  DTypeType::get(buf.getContext())});
        })
        .Case([&](TensorType tensor) {
          auto indexType = IndexType::get(tensor.getContext());
          // Convert tensor types to a struct of the form
          // {
          //    pointer,   --- for buffer
          //    index,     --- for rank
          //    index[5],  --- for shape
          //    dtype      --- for dtype
          // }
          return StructType::get(
              {tensor.getPointerType(), indexType,
               POP::ArrayType::get(TensorType::getMaximumRank(), indexType),
               DTypeType::get(tensor.getContext())});
        })
        .Default([&](Type type) { return type; });
  });

  // Configure dialect conversion
  ConversionTarget target(getContext());
  target.addIllegalDialect<ZAPDialect>();
  target.addLegalDialect<index::IndexDialect, KGENDialect, POPDialect>();

  auto isZAPType = [&](Type type) {
    return type.isa<BufferType, TensorType>();
  };

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
