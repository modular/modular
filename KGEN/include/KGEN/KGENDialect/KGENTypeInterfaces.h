//===- KGEN/KGENDialect/KGENTypeInterfaces.h ------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENTYPEINTERFACES_H
#define KGEN_KGENDIALECT_KGENTYPEINTERFACES_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ML/DType.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Location.h"

//===----------------------------------------------------------------------===//
// OpaqueObjectInterface Utility Functions
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Fill `obj` according to `kind`, `dtype`, and `numElements`. Despite `obj`
/// being suggestively named, `obj` can be any pointer - it does not have to be
/// the pointer passed to OpaqueObjectInterface::populate. It must have a space
/// allocated for `numElements` objects of type `dtype`, however.
///
/// The provided location is used to emit errors in the case the fill fails.
LogicalResult fillOpaqueElements(Location loc, InputGenKind kind, DType dtype,
                                 size_t numElements, void *obj);

/// Compares raw buffers `lhs` and `rhs` of type `dtype` with `numElements`
/// elements. Returns true if they are equal, false if they are not, and failure
/// if they cannot be compared.
///
/// The provided location is used to emit errors in the case the compare fails.
FailureOr<bool> compareOpaqueElements(Location loc, DType dtype,
                                      size_t numElements, void *lhs, void *rhs);
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENTypeInterfaces.h.inc"

#endif // KGEN_KGENDIALECT_KGENTYPEINTERFACES_H
