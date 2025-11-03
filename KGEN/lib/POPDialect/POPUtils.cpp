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
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include <mlir/IR/Builders.h>

using namespace M;
using namespace KGEN;
using namespace POP;

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
                           SIMDType inputType, SIMDType outputType) {
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
    return foldSIMDOpResult<kOtherResult>(
        operands, *dtype,
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
    unsigned width = dtype->getIntegerWidthInBits();
    return foldSIMDOpResult<kOtherResult>(
        operands, *dtype,
        [&](const APSInt &in) -> APSInt { return in.extOrTrunc(width); },
        [&](const APFloat &in) -> std::optional<APSInt> {
          APSInt iv(width, /*isUnsigned=*/dtype->isUInt());
          bool ignored;
          if (in.convertToInteger(iv, APFloat::rmTowardZero, &ignored) ==
              APFloat::opInvalidOp)
            return {};
          return iv;
        },
        [&](bool in) { return APSInt(APInt(width, in), dtype->isUInt()); });
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

  if (size == 1) {
    if (scalarAttr)
      return scalarAttr;
    return scalarVal;
  }

  auto scalarSIMD = dyn_cast_if_present<SIMDAttr>(scalarAttr);
  if (!size || !scalarSIMD)
    return {};
  SmallVector<DTypeValue> values(*size, scalarSIMD.getValues().front());
  return SIMDAttr::get(values, resultType);
}
