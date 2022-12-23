//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the core KGEN attribute classes, provides implementation
// logic for working with them, and helpers for defining operations that take
// them.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENATTRS_H
#define KGEN_KGENDIALECT_KGENATTRS_H

#include "KGEN/KGENDialect/KGENAttrInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace M {
class TargetInfoAttr;

namespace KGEN {
class KGENDType;
class ListType;
class SignatureType;
class TargetType;

//===----------------------------------------------------------------------===//
// TypeConstantAttr
//===----------------------------------------------------------------------===//

/// Base class for MLIR type constant attributes. This attribute represents a
/// constant MLIR type expression.
class TypeConstantAttr : public Attribute {
public:
  using Attribute::Attribute;

  /// Returns the constant type value.
  Type getValue() const;

  /// Get a type constant attribute.
  static TypedAttr get(Type value);

  /// Support type inquiry.
  static bool classof(Attribute attr);
};

} // namespace KGEN
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

// Pull in all enum type definitions and utility function declarations.
#include "KGEN/KGENDialect/KGENEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.h.inc"

//===----------------------------------------------------------------------===//
// PointerLikeTypeTraits
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct PointerLikeTypeTraits<M::KGEN::ParamDeclRefAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline M::KGEN::ParamDeclRefAttr getFromVoidPointer(void *p) {
    return M::KGEN::ParamDeclRefAttr::getFromOpaquePointer(p);
  }
};

template <>
struct PointerLikeTypeTraits<M::KGEN::ParamBindAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline M::KGEN::ParamBindAttr getFromVoidPointer(void *p) {
    return M::KGEN::ParamBindAttr::getFromOpaquePointer(p);
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Return the `paramDecls` array of ParamDeclAttr values if the specified
/// operation has it, or an empty array otherwise.
ArrayRef<ParamDeclAttr> getParamDecls(Operation *op);
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENATTRS_H
