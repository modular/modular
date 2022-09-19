//===- KGEN/MetaDialect/MetaTypes.h ---------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the Meta dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_METADIALECT_METATYPES_H
#define KGEN_METADIALECT_METATYPES_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "Support/ML/DType.h"

#define GET_TYPEDEF_CLASSES
#include "KGEN/MetaDialect/MetaTypes.h.inc"

namespace M::KGEN {

/// Check whether it is valid to cast between a meta type and an MLIR standard
/// type. This function checks whether the data types are compatible using the
/// provided `checkDType` function.
LogicalResult checkMetaCastedTypes(
    function_ref<InFlightDiagnostic(StringRef)> emitError, Type metaTy,
    Type standardTy,
    function_ref<LogicalResult(Type, DTypeConstantAttr)> checkDType);
/// Check whether it is valid to cast between a meta type and an MLIR standard
/// type. This function checks whether the data types are convertible to the
/// given MLIR type.
LogicalResult
checkMetaCastedTypes(function_ref<InFlightDiagnostic(StringRef)> emitError,
                     Type metaTy, Type standardTy);

/// Given a type that implements `DTypeInterface`, return a scalar type of the
/// same dtype as the given type.
ScalarType getScalarOfSameDType(Type type);

} // namespace M::KGEN

#endif // KGEN_METADIALECT_METATYPES_H
