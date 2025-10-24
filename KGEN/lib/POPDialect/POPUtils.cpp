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
