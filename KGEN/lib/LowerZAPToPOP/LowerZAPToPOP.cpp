//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
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
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace POP;
using namespace ZAP;

//===----------------------------------------------------------------------===//
// Type Lowering
//===----------------------------------------------------------------------===//

/// The position of the buffer address in its struct representation.
static constexpr int kBufferAddressPosition = 0;
/// The position of the buffer size in its struct representation.
static constexpr int kBufferSizePosition = 1;
/// The position of the buffer dtype in its struct representation.
static constexpr int kBufferDTypePosition = 2;

/// The position of the ndbuffer address in its struct representation.
static constexpr int kNDBufferAddressPosition = 0;
/// The position of the ndbuffer shape in its struct representation.
static constexpr int kNDBufferShapePosition = 2;
/// The position of the ndbuffer dtype in its struct representation.
static constexpr int kNDBufferDTypePosition = 3;

/// Lower a ZAP type. Passthrough all other types.
static Type convertType(Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case([](BufferType buf) {
        // Convert buffer types to a struct of (pointer, index, dtype).
        return StructType::get({buf.getPointerType(),
                                IndexType::get(buf.getContext()),
                                DTypeType::get(buf.getContext())});
      })
      .Case([](NDBufferType ndBuffer) {
        auto indexType = IndexType::get(ndBuffer.getContext());
        // Convert NDBuffer types to a struct of the form
        // {
        //    pointer,   --- for buffer
        //    index,     --- for rank
        //    index[5],  --- for shape
        //    dtype      --- for dtype
        // }
        return StructType::get(
            {ndBuffer.getPointerType(), indexType,
             POP::ArrayType::get(NDBufferType::getMaximumRank(), indexType),
             DTypeType::get(ndBuffer.getContext())});
      })
      .Default([](Type type) { return type; });
}

/// Materialize a type conversion.
static Value convertValue(Value value) {
  Type type = convertType(value.getType());
  assert(type != value.getType());
  auto b = OpBuilder::atBlockBegin(value.getParentBlock());
  return b.create<mlir::UnrealizedConversionCastOp>(value.getLoc(), type, value)
      .getResult(0);
}

namespace {

//===----------------------------------------------------------------------===//
// ConvertZAPBufferConstruct
//===----------------------------------------------------------------------===//

/// Construct a buffer struct. If either the size or dtype are dynamic, values
/// for them must be provided.
static Value constructBuffer(PatternRewriter &rewriter, Location loc,
                             BufferType type, Value ptr, Value size = {},
                             Value dtype = {}) {
  if (!size)
    size = rewriter.create<ParamConstantOp>(loc, type.getSize());
  if (!dtype)
    dtype = rewriter.create<ParamConstantOp>(loc, type.getDType());
  return rewriter.create<StructConstructOp>(loc, type,
                                            ArrayRef<Value>{ptr, size, dtype});
}

/// Convert the construction of a buffer to building the underlying struct.
struct ConvertZAPBufferConstruct
    : public mlir::OpRewritePattern<BufferConstructOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferConstructOp op,
                                PatternRewriter &rewriter) const override {
    Value buf = constructBuffer(rewriter, op.getLoc(), op.getType(),
                                op.getPtr(), op.getSize(), op.getDType());
    rewriter.replaceOp(op, buf);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferSize
//===----------------------------------------------------------------------===//

/// Extract the buffer size at element 0.
struct ConvertZAPBufferSize : public mlir::OpRewritePattern<BufferSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferSizeOp op,
                                PatternRewriter &rewriter) const override {
    if (TypedAttr size = op.getBuffer().getType().getSize()) {
      rewriter.replaceOpWithNewOp<ParamConstantOp>(op, size);
      return success();
    }
    rewriter.replaceOpWithNewOp<StructGetOp>(op, convertValue(op.getBuffer()),
                                             kBufferSizePosition);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferAddress
//===----------------------------------------------------------------------===//

/// Extract the buffer pointer at element 1.
struct ConvertZAPBufferAddress
    : public mlir::OpRewritePattern<BufferAddressOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferAddressOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<StructGetOp>(op, convertValue(op.getBuffer()),
                                             kBufferAddressPosition);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferDType
//===----------------------------------------------------------------------===//

/// Extract the buffer dtype at element 2.
struct ConvertZAPBufferDType : public mlir::OpRewritePattern<BufferDTypeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferDTypeOp op,
                                PatternRewriter &rewriter) const override {
    if (TypedAttr dtype = op.getBuffer().getType().getDType()) {
      rewriter.replaceOpWithNewOp<ParamConstantOp>(op, dtype);
      return success();
    }
    rewriter.replaceOpWithNewOp<StructGetOp>(op, convertValue(op.getBuffer()),
                                             kBufferDTypePosition);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferBitCast
//===----------------------------------------------------------------------===//

/// When converting an buffer, we have to bitcast the pointer type. Moreover, we
/// will overwrite the size and dtype if they were respecified in the return
/// type.
struct ConvertZAPBufferBitCast
    : public mlir::OpRewritePattern<BufferBitCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferBitCastOp op,
                                PatternRewriter &rewriter) const override {
    BufferType inputType = op.getInput().getType();
    BufferType type = op.getType();
    // If the input and output types are the same, fold away the op.
    if (inputType == type) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }

    // Bitcast the pointer if needed.
    Value popBuf = convertValue(op.getInput());
    Value ptr = rewriter.create<StructGetOp>(op.getLoc(), popBuf,
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
      size = rewriter.create<StructGetOp>(op.getLoc(), popBuf,
                                          kBufferSizePosition);
    if (dtypeExpr)
      dtype = rewriter.create<ParamConstantOp>(op.getLoc(), dtypeExpr);
    else
      dtype = rewriter.create<StructGetOp>(op.getLoc(), popBuf,
                                           kBufferDTypePosition);

    rewriter.replaceOpWithNewOp<StructConstructOp>(
        op, type, ArrayRef<Value>{ptr, size, dtype});
    return success();
  }
};

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
        op.getLoc(), type.getPointerType(), type.getSize());
    Value buf = constructBuffer(rewriter, op.getLoc(), op.getType(), ptr);
    rewriter.replaceOp(op, buf);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPBufferConstant
//===----------------------------------------------------------------------===//

struct ConvertZAPBufferConstant : mlir::OpRewritePattern<BufferConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferConstantOp op,
                                PatternRewriter &rewriter) const override {
    BufferType type = op.getType();
    auto elType = SIMDType::get(1, type.getDType());
    Value global = rewriter.create<GlobalConstantOp>(
        op.getLoc(),
        PointerType::get(POP::ArrayType::get(type.getSize(), elType)),
        op.getValues());
    Value ptr = rewriter.create<PointerBitcastOp>(
        op.getLoc(), PointerType::get(elType), global);
    Value buf = constructBuffer(rewriter, op.getLoc(), type, ptr);
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
    Value base = rewriter.create<StructGetOp>(
        op.getLoc(), convertValue(op.getBuffer()), kBufferAddressPosition);
    Value ptr = rewriter.create<OffsetOp>(op.getLoc(), base, op.getPosition());
    Value bitcastPtr = rewriter.create<PointerBitcastOp>(
        op.getLoc(), PointerType::get(op.getType()), ptr);
    rewriter.replaceOpWithNewOp<LoadOp>(op, op.getType(), bitcastPtr,
                                        op.getAlignmentAttr());
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
    Value base = rewriter.create<StructGetOp>(
        op.getLoc(), convertValue(op.getBuffer()), kBufferAddressPosition);
    Value ptr = rewriter.create<OffsetOp>(op.getLoc(), base, op.getPosition());
    Value bitcastPtr = rewriter.create<PointerBitcastOp>(
        op.getLoc(), PointerType::get(op.getValue().getType()), ptr);
    rewriter.replaceOpWithNewOp<StoreOp>(op, op.getValue(), bitcastPtr,
                                         op.getAlignmentAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPPrint
//===----------------------------------------------------------------------===//

/// Lower the string to a global constant.
static Value lowerStringToGlobalConstant(Operation *op, StringRef str,
                                         OpBuilder &b) {
  auto values = IntArrayElementsAttr::get<char>(
      b.getContext(), {str.data(), str.size()}, IntegerType::Signed);
  auto charType = b.getType<ScalarType>(DType::si8);
  return b.create<GlobalConstantOp>(
      op->getLoc(), PointerType::get(POP::ArrayType::get(str.size(), charType)),
      values);
}

/// Lower the string into a global C string. Null-terminate the string and
/// return an `si8` pointer.
static Value lowerToCString(Operation *op, StringRef str, OpBuilder &b) {
  SmallString<256> nullTerminatedStr = str;
  nullTerminatedStr.push_back('\0');
  auto charType = b.getType<ScalarType>(DType::si8);
  return b.create<PointerBitcastOp>(
      op->getLoc(), PointerType::get(charType),
      lowerStringToGlobalConstant(op, nullTerminatedStr, b));
}

struct ConvertZAPPrint : mlir::OpRewritePattern<PrintOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PrintOp op,
                                PatternRewriter &rewriter) const override {
    // Lower the format into the a global constant.
    Value fmt = lowerToCString(op, op.getFmt(), rewriter);
    // Create the invocation to `printf`.
    SmallVector<Value> operands;
    operands.reserve(op.getNumOperands() + 1);
    operands.push_back(fmt);
    llvm::append_range(operands, op.getOperands());
    rewriter.replaceOpWithNewOp<ExternalCallOp>(
        op, TypeRange(), "printf", operands,
        TypeAttr::get(rewriter.getFunctionType(fmt.getType(), {})));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPGlobalString
//===----------------------------------------------------------------------===//

struct ConvertZAPGlobalString : mlir::OpRewritePattern<GlobalStringOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalStringOp op,
                                PatternRewriter &rewriter) const override {
    Value str = lowerStringToGlobalConstant(op, op.getValue(), rewriter);
    rewriter.replaceOp(op, str);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPDebugAssert
//===----------------------------------------------------------------------===//

struct ConvertZAPDebugAssert : mlir::OpRewritePattern<DebugAssertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DebugAssertOp op,
                                PatternRewriter &rewriter) const override {
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
    Value functionName = lowerToCString(op, functionNameStr, rewriter);
    Value filenameVal = lowerToCString(op, locationStr, rewriter);
    Value message = lowerToCString(op, op.getMsg(), rewriter);

    // Call into the CompilerRT assert function.
    rewriter.replaceOpWithNewOp<ExternalCallOp>(
        op, TypeRange(), "KGEN_CompilerRT_DebugAssert",
        ValueRange{op.getCond(), functionName, filenameVal, message}, nullptr);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPNDBufferConstruct
//===----------------------------------------------------------------------===//

/// Construct a buffer struct. If either the size or dtype are dynamic, values
/// for them must be provided.
static Value constructNDBuffer(PatternRewriter &rewriter, Location loc,
                               NDBufferType type, Value ptr, size_t rank,
                               ValueRange shape = {}, Value dtype = {}) {
  IndexType indexType = rewriter.getIndexType();
  // Initialize the shape values with all zeros.
  auto zeroIndex = rewriter.create<index::ConstantOp>(loc, 0);
  std::array<Value, NDBufferType::getMaximumRank()> shapeValues;
  shapeValues.fill(zeroIndex);
  // Fill the shape values with the provided values or the constants specified
  // by the type.
  for (size_t i = 0, shapeParamOffset = 0; i < rank; ++i) {
    TypedAttr dim = type.getShape()[i];
    // If the dimension is not statically known, then we query the dimension
    // from the user-specified shape values (if available).
    if (!dim) {
      shapeValues[i] = shape[shapeParamOffset++];
      continue;
    }
    // Otherwise, we use the constant value from the type.
    shapeValues[i] =
        TypeSwitch<TypedAttr, Value>(dim)
            .Case([&](IntegerAttr attr) {
              return rewriter.create<index::ConstantOp>(loc, indexType, attr);
            })
            .Case([&](ParamDeclRefAttr attr) {
              return rewriter.create<ParamConstantOp>(loc, attr);
            });
  }
  // Create the shape array.
  auto shapeArray = rewriter.create<ArrayCreateOp>(loc, shapeValues);

  auto rankVal = rewriter.create<index::ConstantOp>(loc, type.getRank());
  if (!dtype)
    dtype = rewriter.create<ParamConstantOp>(loc, type.getDType());
  return rewriter.create<StructConstructOp>(
      loc, type, ValueRange{ptr, rankVal, shapeArray, dtype});
}

/// Convert the construction of a buffer to building the underlying struct.
struct ConvertZAPNDBufferConstruct
    : public mlir::OpRewritePattern<NDBufferConstructOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NDBufferConstructOp op,

                                PatternRewriter &rewriter) const override {
    Value ndbuffer =
        constructNDBuffer(rewriter, op.getLoc(), op.getType(), op.getPtr(),
                          op.getType().getRank(), op.getShape(), op.getDType());
    rewriter.replaceOp(op, ndbuffer);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPNDBufferStackAllocation
//===----------------------------------------------------------------------===//

struct ConvertZAPNDBufferStackAllocation
    : mlir::OpRewritePattern<NDBufferStackAllocationOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NDBufferStackAllocationOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    NDBufferType type = op.getType().cast<NDBufferType>();
    auto size =
        ParamOperatorAttr::get(rewriter.getContext(), POC::Mul, type.getShape(),
                               rewriter.getIndexType());
    Value ptr =
        rewriter.create<StackAllocationOp>(loc, type.getPointerType(), size);
    Value buf = constructNDBuffer(rewriter, loc, type, ptr, type.getRank());
    rewriter.replaceOp(op, buf);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPNDBufferAddress
//===----------------------------------------------------------------------===//

/// Extract the NDBuffer address.
struct ConvertZAPNDBufferAddress
    : public mlir::OpRewritePattern<NDBufferAddressOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NDBufferAddressOp op,

                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<StructGetOp>(op, convertValue(op.getNDBuffer()),
                                             kNDBufferAddressPosition);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPNDBufferRank
//===----------------------------------------------------------------------===//

/// Extract the NDBuffer rank.
struct ConvertZAPNDBufferRank : public mlir::OpRewritePattern<NDBufferRankOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NDBufferRankOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ParamConstantOp>(
        op, rewriter.getIndexAttr(op.getNDBuffer().getType().getRank()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPNDBufferDim
//===----------------------------------------------------------------------===//

/// Extract the buffer dimension at index provided.
struct ConvertZAPNDBufferDim : public mlir::OpRewritePattern<NDBufferDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NDBufferDimOp op,
                                PatternRewriter &rewriter) const override {
    if (TypedAttr dim =
            op.getNDBuffer().getType().getShape()[op.getIndexAttr().getInt()]) {
      rewriter.replaceOpWithNewOp<ParamConstantOp>(op, dim);
      return success();
    }
    Value shape = rewriter.create<StructGetOp>(
        op->getLoc(), convertValue(op.getNDBuffer()), kNDBufferShapePosition);
    rewriter.replaceOpWithNewOp<ArrayGetOp>(op, op.getType(), shape,
                                            op.getIndex());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPNDBufferSize
//===----------------------------------------------------------------------===//

/// If we know the dimension statically, use a constant. Otherwise, query the
/// array.
static Value getDimensionAtIndex(OpBuilder &builder, Location loc,
                                 NDBufferType ndBufferType, Value shape,
                                 size_t idx) {
  ArrayRef<TypedAttr> ndBufferShape = ndBufferType.getShape();
  TypedAttr dim = ndBufferShape[idx];
  if (!dim)
    // Emit op to get dimension from array if not known constant.
    return builder.create<ArrayGetOp>(loc, shape, idx);

  // Use constant or parameter constant if available.
  return TypeSwitch<TypedAttr, Value>(dim)
      .Case([&](IntegerAttr attr) {
        return builder.create<index::ConstantOp>(loc, attr);
      })
      .Case([&](ParamDeclRefAttr attr) {
        return builder.create<ParamConstantOp>(loc, attr);
      });
}

/// Compute the size of the ndBuffer.
struct ConvertZAPNDBufferSize : public mlir::OpRewritePattern<NDBufferSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NDBufferSizeOp op,
                                PatternRewriter &rewriter) const override {
    if (Optional<int64_t> size = op.getNDBuffer().getType().getResolvedSize()) {
      rewriter.replaceOpWithNewOp<ParamConstantOp>(
          op, rewriter.getIndexAttr(*size));
      return success();
    }

    Location loc = op->getLoc();
    NDBufferType ndBufferType = op.getNDBuffer().getType();
    Value shape = rewriter.create<StructGetOp>(
        loc, convertValue(op.getNDBuffer()), kNDBufferShapePosition);
    Value product = getDimensionAtIndex(rewriter, loc, ndBufferType, shape, 0);
    for (size_t i = 1, e = ndBufferType.getRank(); i < e; ++i) {
      Value dim = getDimensionAtIndex(rewriter, loc, ndBufferType, shape, i);
      product = rewriter.create<index::MulOp>(loc, product, dim);
    }
    rewriter.replaceOp(op, product);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPNDBufferDType
//===----------------------------------------------------------------------===//

/// Extract the NDBuffer rank.
struct ConvertZAPNDBufferDType
    : public mlir::OpRewritePattern<NDBufferDTypeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NDBufferDTypeOp op,
                                PatternRewriter &rewriter) const override {
    if (TypedAttr dtype = op.getNDBuffer().getType().getDType()) {
      rewriter.replaceOpWithNewOp<ParamConstantOp>(op, dtype);
      return success();
    }
    rewriter.replaceOpWithNewOp<StructGetOp>(op, convertValue(op.getNDBuffer()),
                                             kNDBufferDTypePosition);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPNDBufferLoad
//===----------------------------------------------------------------------===//

/// Generates pop/index ops that computes the linearized index expression given
/// a NDBuffer shape array and an index array, by essentially computing a dot
/// product between the index and shape vectors. Assumes that the NDBuffer is in
/// row-major contiguous layout. This function also assumes the index and shape
/// layout as [x,y,z,0,0], with x being the highest order dimension i.e.
/// outermost dimension.
///
/// This function computes the dot product between the index and shape vector.
/// Example:
///
///   zap.ndbuffer.load %ndBuffer[x, y, z] : !zap.ndbuffer<[a,b,c], dtype>
///
/// will compute the `accumulatedOffset` by
///
///   accumulatedOffset = z + c*(y + b*x)
///
/// in 2 iterations, each iteration multiplies a number from the shape list and
/// adds a number from the index list.
static Value linearizeContiguousIndices(PatternRewriter &rewriter, Location loc,
                                        NDBufferType ndBufferType,
                                        Value shapeArray,
                                        ValueRange indexArray) {
  // Initialize `accumulatedOffset` with the innermost index. e.g. add the `x`
  // term in the example above.
  Value accumulatedOffset = *indexArray.begin();

  // Initialize the index used to access the shape array.
  // Start at position one as the highest order term of the shape array is not
  // used in computing indices. See e.g. above, term `a` is skipped.
  size_t indexPosition = 1;

  // Iterate through indices and create the multiply and add in each iteration.
  for (auto indexValue : llvm::drop_begin(indexArray)) {
    // Dimension size at current position from the shape list. e.g. load the `b`
    // term from example above.
    Value positionSize = getDimensionAtIndex(rewriter, loc, ndBufferType,
                                             shapeArray, indexPosition++);

    // Multiply by the current size, e.g. x-> b*x from example above.
    accumulatedOffset =
        rewriter.create<index::MulOp>(loc, accumulatedOffset, positionSize);

    // Add the current index, e.g. b*x -> b*x + y from example above.
    accumulatedOffset =
        rewriter.create<index::AddOp>(loc, accumulatedOffset, indexValue);
  }

  return accumulatedOffset;
}

/// Load a simd value from NDBuffer given a list of position indices.
struct ConvertZAPNDBufferLoad : public mlir::OpRewritePattern<NDBufferLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NDBufferLoadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value popBuf = convertValue(op.getNDBuffer());
    Value shapeArray = rewriter.create<StructGetOp>(op->getLoc(), popBuf,
                                                    kNDBufferShapePosition);
    Value base = rewriter.create<StructGetOp>(op.getLoc(), popBuf,
                                              kNDBufferAddressPosition);
    Value offset = linearizeContiguousIndices(
        rewriter, loc, op.getNDBuffer().getType().cast<NDBufferType>(),
        shapeArray, op.getPositions());
    Value ptr = rewriter.create<OffsetOp>(loc, base, offset);
    Value simdPtr = rewriter.create<PointerBitcastOp>(
        loc, PointerType::get(op.getType()), ptr);
    rewriter.replaceOpWithNewOp<LoadOp>(op, op.getType(), simdPtr,
                                        op.getAlignmentAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPNDBufferStore
//===----------------------------------------------------------------------===//

/// Store a simd value into the NDBuffer at the given a list of position
/// indices.
struct ConvertZAPNDBufferStore
    : public mlir::OpRewritePattern<NDBufferStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NDBufferStoreOp op,

                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value popBuf = convertValue(op.getNDBuffer());
    Value shapeArray = rewriter.create<StructGetOp>(op->getLoc(), popBuf,
                                                    kNDBufferShapePosition);
    Value base = rewriter.create<StructGetOp>(op.getLoc(), popBuf,
                                              kNDBufferAddressPosition);
    Value offset = linearizeContiguousIndices(
        rewriter, loc, op.getNDBuffer().getType().cast<NDBufferType>(),
        shapeArray, op.getPositions());
    Value ptr = rewriter.create<OffsetOp>(loc, base, offset);
    Value simdPtr = rewriter.create<PointerBitcastOp>(
        loc, PointerType::get(op.getValue().getType()), ptr);
    rewriter.replaceOpWithNewOp<StoreOp>(op, op.getValue(), simdPtr,
                                         op.getAlignmentAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertZAPNDBufferBitCast
//===----------------------------------------------------------------------===//

struct ConvertZAPNDBufferBitCast
    : public mlir::OpRewritePattern<NDBufferBitCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NDBufferBitCastOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = op.getInput().getType().cast<NDBufferType>();
    auto type = op.getType().cast<NDBufferType>();

    // Bitcast the pointer if needed.
    Value ndBuffer = convertValue(op.getInput());
    Value ptr = rewriter.create<StructGetOp>(op.getLoc(), ndBuffer,
                                             kNDBufferAddressPosition);
    if (type.getDType() != inputType.getDType())
      ptr = rewriter.create<PointerBitcastOp>(
          op.getLoc(), op.getType().getPointerType(), ptr);

    // Query the source ndbuffer for the dynamic shape information.
    Value shapeArray;
    SmallVector<Value, 4> dynamicShapeValues;
    for (auto [idx, shape] : llvm::enumerate(type.getShape())) {
      if (shape)
        continue;
      if (!shapeArray)
        shapeArray = rewriter.create<StructGetOp>(op.getLoc(), ndBuffer,
                                                  kNDBufferShapePosition);
      dynamicShapeValues.emplace_back(
          rewriter.create<ArrayGetOp>(op.getLoc(), shapeArray, idx));
    }

    rewriter.replaceOp(op,
                       constructNDBuffer(rewriter, op.getLoc(), type, ptr,
                                         type.getRank(), dynamicShapeValues));
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateZAPToPOPPatterns(RewritePatternSet &patterns) {
  patterns.insert<
      // clang-format off
      ConvertZAPBufferAddress,
      ConvertZAPBufferBitCast,
      ConvertZAPBufferConstant,
      ConvertZAPBufferConstruct,
      ConvertZAPBufferDType,
      ConvertZAPBufferLoad,
      ConvertZAPBufferSize,
      ConvertZAPBufferStackAllocation,
      ConvertZAPBufferStore,
      ConvertZAPDebugAssert,
      ConvertZAPGlobalString,
      ConvertZAPNDBufferAddress,
      ConvertZAPNDBufferBitCast,
      ConvertZAPNDBufferConstruct,
      ConvertZAPNDBufferDType,
      ConvertZAPNDBufferDim,
      ConvertZAPNDBufferLoad,
      ConvertZAPNDBufferRank,
      ConvertZAPNDBufferSize,
      ConvertZAPNDBufferStackAllocation,
      ConvertZAPNDBufferStore,
      ConvertZAPPrint
      // clang-format on
      >(patterns.getContext());
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

  // Populate patterns
  RewritePatternSet patterns(&getContext());
  populateZAPToPOPPatterns(patterns);
  mlir::FrozenRewritePatternSet set(std::move(patterns));
  mlir::PatternApplicator applicator(set);
  applicator.applyDefaultCostModel();

  // Collect all ops to rewrite.
  std::vector<Operation *> opsToRewrite;
  Dialect *zapDialect = getContext().getLoadedDialect<ZAPDialect>();
  theModule.walk([&](Operation *op) {
    if (op->getDialect() == zapDialect)
      opsToRewrite.push_back(op);
  });

  // Run the conversion.
  struct SimplePatternRewriter : public PatternRewriter {
    explicit SimplePatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
  };
  SimplePatternRewriter rewriter(&getContext());
  for (Operation *op : opsToRewrite) {
    if (failed(applicator.matchAndRewrite(op, rewriter))) {
      op->emitError("failed to lower ZAP operation");
      return signalPassFailure();
    }
  }

  // Lower all ZAP remaining types.
  auto convertNestedTypes = [](Type type) {
    if (auto itf = dyn_cast<mlir::SubElementTypeInterface>(type))
      return convertType(itf.replaceSubElements(convertType));
    return convertType(type);
  };
  theModule.walk([&](Operation *op) {
    op->setAttrs(cast<DictionaryAttr>(
        op->getAttrDictionary().replaceSubElements(convertType)));
    for (Value value : op->getResults())
      value.setType(convertNestedTypes(value.getType()));
    for (Region &region : op->getRegions())
      for (Value value : region.getArguments())
        value.setType(convertNestedTypes(value.getType()));
    if (auto cast = dyn_cast<mlir::UnrealizedConversionCastOp>(op))
      if (llvm::any_of(cast.getOperandTypes(), [&](Type type) {
            return &type.getDialect() == zapDialect;
          }))
        rewriter.replaceOp(cast, cast.getInputs());
  });
}
