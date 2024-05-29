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
#include "KGEN/KGENDialect/KGENEnums.h"
#include "Support/ErrorOr.h"
#include "Support/IPInt.h"
#include "Support/IPRational.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
class OperationName;
} // namespace mlir

namespace M {
class TargetInfoAttr;

namespace KGEN {
class BuildInfoType;
class KGENDType;
class SignatureType;
class TargetType;
class VariadicType;
class VariadicAttr;
class VTableAttr;

//===----------------------------------------------------------------------===//
// TypeConstantAttr
//===----------------------------------------------------------------------===//

/// Base class for MLIR type constant attributes. This attribute represents a
/// constant MLIR type expression.
class TypeConstantAttr : public Attribute {
public:
  using Attribute::Attribute;

  /// Returns the type-representation of this constant type value.
  Type getMlirType() const;

  /// Get the metatype.
  Type getType() const;

  /// Returns the constant type vtable.
  VTableAttr getVTable() const;

  /// Get a type constant attribute.
  static TypedAttr get(Type mlirType, Type type);
  /// Get a type constant attribute with a vtable.
  static TypedAttr get(Type mlirType, Type type, VTableAttr vtable);

  /// Returns true if the given type is classified as a concrete type.
  static bool isConcreteType(Type type);

  /// Support type inquiry.
  static bool classof(Attribute attr);
};

} // namespace KGEN
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Attribute Declarations
//===----------------------------------------------------------------------===//

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
} // namespace llvm

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Emit an MLIR operation call in a parameter context.
TypedAttr emitMLIROperationCall(
    StringRef opName,
    ArrayRef<std::pair<StringAttr (*)(mlir::OperationName), Attribute>> attrs,
    ArrayRef<TypedAttr> operands, Type resultType);
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENATTRS_H
