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

#include "KGEN/KGENDialect/KGENDType.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

// Pull in all enum type definitions and utility function declarations.
#include "KGEN/KGENDialect/KGENEnums.h.inc"

namespace M::KGEN {
class ConcreteTypeConstantAttr;
class ConstraintAttr;
class DTypeConstantAttr;
class ParamDeclArrayAttr;
class ParamDeclAttr;
class ParameterizedTypeConstantAttr;
class SignatureType;
class SymbolConstantAttr;

/// Return the `paramDecls` array of ParamDeclAttr values if the specified
/// operation has it, or an empty array otherwise.
ArrayRef<ParamDeclAttr> getParamDecls(Operation *op);

/// We expect all parameter expressions to simplify down to concrete constants
/// after elaboration.  We don't want anything left as a ParamOperatorAttr or
/// ParamDeclRefAttr or ParameterizedTypeConstantAttr.
inline bool isSimpleConstant(Attribute attr) {
  return attr.isa<FloatAttr, IntegerAttr, StringAttr, DTypeConstantAttr,
                  ConcreteTypeConstantAttr, SymbolConstantAttr>();
}

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

} // namespace M::KGEN

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.h.inc"

#endif // KGEN_KGENDIALECT_KGENATTRS_H
