//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions primarily for parsing, printing and
// verifying POP related operations and types.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPUtils.h"
#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/Interpreter/InterpreterState.h"
#include "KGEN/Interpreter/ParametricInterpreterState.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace KGEN;
using namespace POP;

/// Get the value of a scalar index-like parameter value.
/// This is a temporary helper utility during the Int->SIMD unification project.
/// After it's done, we should remove the IntegerAttr case.
ErrorOr<int64_t> POP::getScalarIndexValue(TypedAttr value) {
  if (auto intAttr = dyn_cast<IntegerAttr>(value))
    return intAttr.getInt();
  if (auto simdAttr = dyn_cast<POP::SIMDAttr>(value)) {
    ArrayRef<DTypeValue> values = simdAttr.getValues();
    if (values.size() != 1)
      return Error("expected a scalar SIMD value");
    if (!values.front().getDType().isIndex())
      return Error("expected an index-typed SIMD value");
    return values.front().getIndexVal();
  }
  return Error("expected an integer or scalar SIMD");
}

/// Verify a conversion between a SIMD type and an MLIR builtin type.
/// Conversions are assumed to be bi-directional. In error messages, the
/// direction of the conversion is controlled by the `fromSimd` parameter.
LogicalResult
POP::verifyConversionCast(function_ref<InFlightDiagnostic(StringRef)> emitError,
                          SIMDType simd, Type builtinType, bool fromSimd) {
  // Verify the SIMD size matches the vector size and the dtypes match.
  auto size = simd.getResolvedSize();
  if (size && *size == 1) {
    // Scalar case
    auto dtype = dyn_cast<DTypeConstantAttr>(simd.getDType());
    if (dtype && !dtype.isConvertibleTo(builtinType))
      return emitError("cannot convert ")
             << (fromSimd ? "from" : "to") << " scalar dtype "
             << dtype.getDType().getAsString() << (fromSimd ? " to " : " from ")
             << builtinType;
    return success();
  }

  auto vector = dyn_cast<VectorType>(builtinType);
  if (!vector || vector.getRank() != 1 || vector.isScalable())
    return emitError("expected a rank 1 non-scalable vector");

  if (size && *size != vector.getShape().front())
    return emitError("expected vector<") << *size << "xT>";

  if (auto dtype = dyn_cast<DTypeConstantAttr>(simd.getDType());
      dtype && !dtype.isConvertibleTo(vector.getElementType()))
    return emitError("cannot convert ")
           << (fromSimd ? "from" : "to") << " SIMD dtype "
           << dtype.getDType().getAsString() << (fromSimd ? " to" : " from")
           << " vector element " << vector.getElementType();
  return success();
}

/// Convert a SIMD attribute to a vector-typed attribute.
template <typename AttrT, typename TransformFn>
static ArrayElementsAttr convertSIMDToVectorAttr(SIMDAttr simd, VectorType type,
                                                 TransformFn fn) {
  SmallVector<decltype(fn(std::declval<DTypeValue>()))> values;
  for (const DTypeValue &value : simd.getValues())
    values.push_back(fn(value));
  return AttrT::get(type, values);
}

OpFoldResult POP::foldCastToBuiltin(TypedAttr input, Type resultType) {
  if (auto cast = dyn_cast_if_present<CastFromBuiltinAttr>(input))
    if (cast.getArg().getType() == resultType)
      return cast.getArg();

  auto simd = dyn_cast_if_present<POP::SIMDAttr>(input);
  if (!simd)
    return {};
  // Conversion to a 1D vector type.
  std::optional<KGENDType> dtype = simd.getType().getResolvedDType();
  if (!dtype)
    return {};

  if (auto vector = dyn_cast<VectorType>(resultType)) {
    if (dtype->isBool())
      return convertSIMDToVectorAttr<IntArrayElementsAttr>(
          simd, vector,
          [](DTypeValue simd) { return APInt(1, simd.getBoolVal()); });
    if (dtype->isIndex() || dtype->isUIndex())
      return convertSIMDToVectorAttr<IndexArrayElementsAttr>(
          simd, vector, [](DTypeValue simd) { return simd.getIndexVal(); });
    if (dtype->isInt())
      return convertSIMDToVectorAttr<IntArrayElementsAttr>(
          simd, vector, [](DTypeValue simd) { return simd.getIntVal(); });
    assert(dtype->isFloat() && "unexpected dtype");
    return convertSIMDToVectorAttr<FloatArrayElementsAttr>(
        simd, vector, [](DTypeValue simd) { return simd.getFloatVal(); });
  }

  assert(simd.getValues().size() == 1 && "expected a scalar constant");
  const DTypeValue &value = simd.getValues().front();

  // Convert to a scalar attribute.
  Builder b(simd.getContext());
  if (dtype->isBool())
    return b.getBoolAttr(value.getBoolVal());
  if (dtype->isIndex() || dtype->isUIndex())
    return b.getIndexAttr(value.getIndexVal());
  if (dtype->isInt())
    return b.getIntegerAttr(cast<IntegerType>(resultType), value.getIntVal());
  assert(dtype->isFloat() && "unexpected dtype");
  return b.getFloatAttr(cast<FloatType>(resultType), value.getFloatVal());
}

OpFoldResult POP::foldCastFromBuiltin(TypedAttr val, SIMDType resultType) {
  if (auto cast = dyn_cast_if_present<CastToBuiltinAttr>(val))
    if (cast.getArg().getType() == resultType)
      return cast.getArg();

  // Ensure the incoming value is an expected constant kind.
  if (!isa<IntArrayElementsAttr, FloatArrayElementsAttr, IndexArrayElementsAttr,
           IntegerAttr, FloatAttr>(val))
    return {};

  // Conversion from vector constant.
  std::optional<KGENDType> dtype = resultType.getResolvedDType();
  if (!dtype)
    return {};
  if (auto vector = dyn_cast<VectorType>(val.getType())) {
    SmallVector<DTypeValue> values;
    if (dtype->isBool())
      for (APInt value : cast<IntArrayElementsAttr>(val).getValues())
        values.emplace_back(!value.isZero(), *dtype);
    else if (dtype->isIndex() || dtype->isUIndex())
      for (int64_t value : cast<IndexArrayElementsAttr>(val))
        values.emplace_back(value, *dtype);
    else if (dtype->isInt())
      for (APInt value : cast<IntArrayElementsAttr>(val).getValues())
        values.emplace_back(value, *dtype);
    else
      for (APFloat value : cast<FloatArrayElementsAttr>(val).getValues())
        values.emplace_back(value, *dtype);
    return SIMDAttr::get(values, resultType);
  }

  // Handle scalar constants.
  if (dtype->isBool())
    return SIMDAttr::get({cast<BoolAttr>(val).getValue(), *dtype}, resultType);
  if (dtype->isIndex() || dtype->isUIndex())
    return SIMDAttr::get({cast<IntegerAttr>(val).getInt(), *dtype}, resultType);
  if (dtype->isInt())
    return SIMDAttr::get({cast<IntegerAttr>(val).getValue(), *dtype},
                         resultType);
  assert(dtype->isFloat() && "unexpected dtype");
  return SIMDAttr::get({cast<FloatAttr>(val).getValue(), *dtype}, resultType);
}

/// Fold a cast between two SIMD types.
OpFoldResult POP::foldCast(ArrayRef<Attribute> operands, SIMDType resultType,
                           SIMDType inputType, SIMDType outputType,
                           std::optional<int64_t> indexBitWidth) {
  auto in = dyn_cast_if_present<SIMDAttr>(operands.front());
  std::optional<KGENDType> dtype = resultType.getResolvedDType();

  if (!in || !dtype) {
    if (inputType == outputType)
      return operands.front();
    return {};
  }

  std::optional<KGENDType> inType = in.getType().getResolvedDType();

  // Exit early if the input and output dtypes are the same.
  if (*dtype == *inType)
    return in;

  if (dtype->isFloat()) {
    // Cannot fold cast to unsupported float dtype.
    const llvm::fltSemantics *sem = dtype->getFloatSemantics();
    if (!sem)
      return {};
    return foldSIMDOpResult<POP::kOtherResult>(
        operands, *dtype, indexBitWidth,
        [&](const APSInt &in) -> APFloat {
          APFloat fp(*sem);
          fp.convertFromAPInt(in, in.isSigned(), APFloat::rmNearestTiesToEven);
          return fp;
        },
        [&](APFloat in) {
          bool ignored;
          in.convert(*sem, APFloat::rmNearestTiesToEven, &ignored);
          return in;
        },
        [&](bool in) { return APFloat(*sem, in); });
  }

  if (dtype->isInt()) {
    // Note that float to integer casts are undefined if the float value is
    // too large to fit in the integer dtype.
    unsigned intWidth = dtype->getIntegerWidthInBits();
    return foldSIMDOpResult<POP::kOtherResult>(
        operands, *dtype, indexBitWidth,
        [&](const APSInt &in) -> APSInt { return in.extOrTrunc(intWidth); },
        [&](const APFloat &in) -> std::optional<APSInt> {
          APSInt iv(intWidth, /*isUnsigned=*/dtype->isUInt());
          bool ignored;
          if (in.convertToInteger(iv, APFloat::rmTowardZero, &ignored) ==
              APFloat::opInvalidOp)
            return {};
          return iv;
        },
        [&](bool in) { return APSInt(APInt(intWidth, in), dtype->isUInt()); });
  }

  if (dtype->isIndex() || dtype->isUIndex() || dtype->isAddress()) {
    // The folding infra ensures that platform-dependent types are only folded
    // if the folding result is the same on 32 and 64 bit platforms. This is the
    // right thing to do for most operations, but casting can be safely allowed
    // between platform-dependent types.
    if (inType->isIndex() || inType->isUIndex() || inType->isAddress()) {
      return Detail::foldSIMDOpImpl(
          std::make_index_sequence<1>(), operands,
          [inType](const APSInt &in) -> int64_t {
            return inType->isSInt() ? in.getSExtValue() : in.getZExtValue();
          },
          *dtype,
          [](DTypeValue val) {
            return APSInt(APInt(64, val.getIndexVal()),
                          /*isUnsigned=*/!val.getDType().isIndex());
          });
    }

    // Cast to index like it's a 64-bit integer. Address is handled like index.
    return foldSIMDOpResult<kOtherResult>(
        operands, *dtype,
        [inType](const APSInt &in) -> int64_t {
          if (in.getSignificantBits() > 64) {
            auto truncated = in.trunc(64);
            return inType->isSInt() ? truncated.getSExtValue()
                                    : truncated.getZExtValue();
          }
          return inType->isSInt() ? in.getSExtValue() : in.getZExtValue();
        },
        [](const APFloat &in) -> std::optional<int64_t> {
          APSInt iv(64, /*isUnsigned=*/false);
          bool ignored;
          if (in.convertToInteger(iv, APFloat::rmTowardZero, &ignored) ==
              APFloat::opInvalidOp)
            return {};
          return iv.getSExtValue();
        },
        [](bool in) { return static_cast<int64_t>(in); });
  }

  if (dtype->isInvalid()) {
    // Invalid is not inhabitable.
    return {};
  }

  assert(dtype->isBool());
  return foldSIMDOpResult<kOtherResult>(
      operands, *dtype, [](const APSInt &in) -> bool { return !in.isZero(); },
      [](const APFloat &in) -> bool { return !in.isZero(); });
}

OpFoldResult POP::foldSIMDSplat(Value scalarVal, Attribute scalarAttr,
                                SIMDType resultType) {
  std::optional<int64_t> size = resultType.getResolvedSize();

  if (size == 1)
    return scalarAttr ? OpFoldResult(scalarAttr) : scalarVal;

  auto scalarSIMD = dyn_cast_if_present<SIMDAttr>(scalarAttr);
  if (!size || !scalarSIMD)
    return {};
  SmallVector<DTypeValue> values(*size, scalarSIMD.getValues().front());
  return SIMDAttr::get(values, resultType);
}

/// Fold a SIMD Or-reduction operation.
OpFoldResult POP::foldSIMDReduceOr(Value vectorVal, Attribute vectorAttr,
                                   SIMDType vectorType) {
  // If the vector only has one element, it's already reduced.
  if (vectorType.getResolvedSize() == 1)
    return vectorAttr ? OpFoldResult(vectorAttr) : vectorVal;

  if (auto dtype = vectorType.getResolvedDType(); !dtype || !dtype->isIntLike())
    return {};

  return foldBitwiseSIMDReduceOp(
      vectorAttr, [](APSInt lhs, APSInt rhs) { return lhs | rhs; },
      [](bool lhs, bool rhs) { return lhs | rhs; });
}

/// Fold a SIMD And-reduction operation.
OpFoldResult POP::foldSIMDReduceAnd(Value vectorVal, Attribute vectorAttr,
                                    SIMDType vectorType) {
  // If the vector only has one element, it's already reduced.
  if (vectorType.getResolvedSize() == 1)
    return vectorAttr ? OpFoldResult(vectorAttr) : vectorVal;

  if (auto dtype = vectorType.getResolvedDType(); !dtype || !dtype->isIntLike())
    return {};

  return foldBitwiseSIMDReduceOp(
      vectorAttr, [](APSInt lhs, APSInt rhs) { return lhs & rhs; },
      [](bool lhs, bool rhs) { return lhs & rhs; });
}

OpFoldResult POP::foldSIMDShl(Attribute val, Attribute shft,
                              TargetInfoAttr targetInfo) {
  auto valSIMD = dyn_cast_if_present<SIMDAttr>(val);
  if (!valSIMD || !isa_and_present<SIMDAttr>(shft))
    return {};
  std::optional<KGENDType> dtype = valSIMD.getType().getResolvedDType();
  if (!dtype)
    return {};

  std::optional<int64_t> indexBitWidth;
  if (targetInfo)
    indexBitWidth = targetInfo.resolveIndexBitWidth();

  return foldSIMDOp({val, shft}, indexBitWidth,
                    [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
                      if (rhs.uge(lhs.getBitWidth()))
                        return std::nullopt;
                      return APSInt(lhs.shl(rhs), !lhs.isSigned());
                    });
}

OpFoldResult POP::foldSIMDShr(Attribute val, Attribute shft,
                              TargetInfoAttr targetInfo) {
  auto valSIMD = dyn_cast_if_present<SIMDAttr>(val);
  if (!valSIMD || !isa_and_present<SIMDAttr>(shft))
    return {};
  std::optional<KGENDType> dtype = valSIMD.getType().getResolvedDType();
  if (!dtype)
    return {};

  std::optional<int64_t> indexBitWidth;
  if (targetInfo)
    indexBitWidth = targetInfo.resolveIndexBitWidth();

  return foldSIMDOp({val, shft}, indexBitWidth,
                    [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
                      if (rhs.uge(lhs.getBitWidth()))
                        return std::nullopt;
                      return APSInt(lhs.isSigned() ? lhs.ashr(rhs)
                                                   : lhs.lshr(rhs),
                                    !lhs.isSigned());
                    });
}

OpFoldResult POP::foldSIMDAbs(Attribute operand, TargetInfoAttr targetInfo) {
  auto operandSIMD = dyn_cast_if_present<SIMDAttr>(operand);
  if (!operandSIMD)
    return {};
  std::optional<KGENDType> dtype = operandSIMD.getType().getResolvedDType();
  if (!dtype || dtype->isInvalid())
    return {};

  // Bools or unsigned types are already abs'd.
  if (dtype->isBool() || dtype->isUInt())
    return operand;

  std::optional<int64_t> indexBitWidth;
  if (targetInfo)
    indexBitWidth = targetInfo.resolveIndexBitWidth();

  return foldSIMDOp(
      operand, indexBitWidth,
      [](APFloat operand) -> APFloat {
        operand.clearSign();
        return operand;
      },
      [](APSInt operand) -> std::optional<APSInt> {
        return APSInt(operand.abs(), /*isUnsigned=*/false);
      });
}

OpFoldResult POP::foldSIMDRound(Attribute operand, TargetInfoAttr targetInfo) {
  auto operandSIMD = dyn_cast_if_present<SIMDAttr>(operand);
  if (!operandSIMD)
    return {};
  std::optional<KGENDType> dtype = operandSIMD.getType().getResolvedDType();
  if (!dtype || dtype->isInvalid())
    return {};

  // Anything that isn't a floating-point value is already rounded
  if (!dtype->isFloat())
    return operand;

  return foldSIMDOp(operand, [](APFloat operand) -> APFloat {
    operand.roundToIntegral(APFloat::rmNearestTiesToEven);
    return operand;
  });
}

OpFoldResult POP::foldSIMDFloorDiv(Attribute lhs, Attribute rhs,
                                   TargetInfoAttr targetInfo) {
  auto lhsSIMD = dyn_cast_if_present<SIMDAttr>(lhs);
  if (!lhsSIMD || !isa_and_present<SIMDAttr>(rhs))
    return {};
  std::optional<KGENDType> dtype = lhsSIMD.getType().getResolvedDType();
  if (!dtype || dtype->isBool() || dtype->isInvalid())
    return {};

  std::optional<int64_t> indexBitWidth;
  if (targetInfo)
    indexBitWidth = targetInfo.resolveIndexBitWidth();

  return foldSIMDOp(
      {lhs, rhs}, indexBitWidth,
      [](APFloat lhs, APFloat rhs) -> APFloat {
        auto div = lhs / rhs;
        div.roundToIntegral(APFloat::rmTowardNegative);
        return div;
      },
      [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
        // Integer division by zero is UB - don't fold.
        if (rhs.isZero())
          return std::nullopt;
        APSInt div = lhs / rhs;
        if (lhs.isUnsigned())
          return div;
        // int div = lhs / rhs;
        // return div * rhs == lhs ? div : div - ((lhs < 0) ^ (rhs < 0));
        int xorOp = (lhs < 0) ^ (rhs < 0);
        return APSInt(div * rhs == lhs ? div : div - xorOp,
                      /*isUnsigned=*/false);
      });
}

template <typename State>
static ErrorTreeOrSuccess interpretMemcpy(Attribute dst, Attribute src,
                                          Attribute len, Location loc,
                                          State &state) {
  auto lenAttr = dyn_cast<IntegerAttr>(len);
  if (!lenAttr)
    return ErrorTree(loc, "interpreting memcpy 3nd operand len is not "
                          "interpreted correctly");

  if (!lenAttr.getInt())
    return success();

  auto dstPtr = dyn_cast<M::PointerAttr>(dst);
  auto srcPtr = dyn_cast<M::PointerAttr>(src);

  if (!dstPtr)
    return ErrorTree(loc, "interpreting memcpy 1st operand dst addr is "
                          "not interpreted correctly");
  if (!srcPtr)
    return ErrorTree(loc, "interpreting memcpy 2nd operand src addr is "
                          "not interpreted correctly");

  ErrorOr<void *> dstAddrOr =
      state.getWritableMemory(dstPtr.getAddr(), size_t(lenAttr.getInt()));
  ErrorOr<const void *> srcAddrOr =
      state.getReadableMemory(srcPtr.getAddr(), size_t(lenAttr.getInt()));

  if (dstAddrOr.isError())
    return ErrorTree(
        loc, "interpreting memcpy can't get dst memory from the interpreter");

  if (srcAddrOr.isError())
    return ErrorTree(
        loc, "interpreting memcpy can't get src memory from the interpreter");

  std::memcpy(*dstAddrOr, *srcAddrOr, lenAttr.getInt());

  return success();
}

ErrorTreeOrSuccess POP::interpretMemcpy(Attribute dst, Attribute src,
                                        Attribute len, Location loc,
                                        InterpreterState &state) {
  return ::interpretMemcpy<InterpreterState>(dst, src, len, loc, state);
}

ErrorTreeOrSuccess POP::interpretMemcpy(Attribute dst, Attribute src,
                                        Attribute len, Location loc,
                                        ParametricInterpreterState &state) {
  return ::interpretMemcpy<ParametricInterpreterState>(dst, src, len, loc,
                                                       state);
}
