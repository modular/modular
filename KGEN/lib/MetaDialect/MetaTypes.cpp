//===- MetaTypes.cpp ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MetaDialect/MetaTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

using namespace M;
using namespace KGEN;

LogicalResult M::KGEN::checkMetaCastedTypes(
    function_ref<InFlightDiagnostic(StringRef)> emitError, Type metaTy,
    Type standardTy,
    function_ref<LogicalResult(DTypeConstantAttr)> checkDType) {
  if (auto scalarTy = metaTy.dyn_cast<ScalarType>()) {
    // Check that the data types match.
    if (auto dtype = scalarTy.getDType().dyn_cast<DTypeConstantAttr>();
        dtype && failed(checkDType(dtype)))
      return emitError("incompatible scalar data type");
    return success();
  }

  // Check that the standard type is a rank 1 vector with matching dimensions.
  auto simdTy = metaTy.cast<SIMDType>();
  auto vectorTy = standardTy.dyn_cast<VectorType>();
  if (!vectorTy)
    return emitError("expected a vector type");
  if (vectorTy.getNumScalableDims() != 0)
    return emitError("vector type should not be scalable");
  if (vectorTy.getRank() != 1)
    return emitError("expected a rank 1 vector");
  if (auto size = simdTy.getSize().dyn_cast<IntegerAttr>();
      size.getInt() != vectorTy.getShape().front())
    return emitError("dimensions do not match");
  if (auto dtype = simdTy.getDType().dyn_cast<DTypeConstantAttr>();
      dtype && failed(checkDType(dtype)))
    return emitError("element types do not match");
  return success();
}

LogicalResult M::KGEN::checkMetaCastedTypes(
    function_ref<InFlightDiagnostic(StringRef)> emitError, Type metaTy,
    Type standardTy) {
  return checkMetaCastedTypes(
      emitError, metaTy, standardTy, [standardTy](DTypeConstantAttr dtype) {
        return success(dtype.isCompatibleWith(standardTy));
      });
}
