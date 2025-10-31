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
