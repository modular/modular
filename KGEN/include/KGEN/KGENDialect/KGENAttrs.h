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
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
class OperationName;
} // namespace mlir

namespace M {
class TargetInfoAttr;

namespace KGEN {
class BuildInfoType;
class ConformanceOp;
class FuncOp;
class GeneratorOp;
class KGENDType;
class ParameterEvaluator;
class TargetType;
class ParamListType;
class VariadicAttr;
} // namespace KGEN
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Attribute Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.h.inc"

//===----------------------------------------------------------------------===//
// EmitAsAttr
//===----------------------------------------------------------------------===//

namespace M::KGEN {
class EmitAsAttr : public IntegerAttr {
public:
  using IntegerAttr::IntegerAttr;
  static bool classof(Attribute attr);
  static EmitAsAttr get(MLIRContext *ctx, EmitAs val);
  EmitAs getValue() const;
};

//===----------------------------------------------------------------------===//
// Sugar Processing for Type and Attribute
//===----------------------------------------------------------------------===//
//
// SugarAttr represents a "syntax sugar" on a type or typed attr, e.g. when
// resolving "alias four = 4", we might want to preserve the name "four" instead
// of inlining the value.  These are typically looked through by semantic
// analysis, but used when generating user-visible error messages.
//
// SugarAttr is lowered away by LowerLIT.

/// Given an attribute or type, return the "canonical" version of the attribute
/// with all type sugar removed.
Attribute getCanonicalAttr(Attribute src);
TypedAttr getCanonicalAttr(TypedAttr src);
Type getCanonicalType(Type type);

/// Return true if the specified types are canonically equal.
bool isEqualCanon(Type t1, Type t2);
bool isEqualCanon(TypedAttr ta1, TypedAttr ta2);

template <typename T>
constexpr bool isValidSugarCastType =
    (std::is_convertible_v<T, Attribute> || std::is_convertible_v<T, Type>);

// Helpers for sugar-aware casting.
template <typename... To, typename From>
[[nodiscard]] inline bool sugarIsa(From val) {
  static_assert(isValidSugarCastType<From>,
                "sugared casts only work with Type and Attribute");
  auto stripped = SugarAttr::strip(val);
  return (isa<To>(stripped) || ...);
}

// Helpers for sugar-aware casting.
template <typename... To, typename From>
[[nodiscard]] inline bool sugarIsaAndNonNull(From val) {
  if (!val)
    return false;

  static_assert(isValidSugarCastType<From>,
                "sugared casts only work with Type and Attribute");
  val = SugarAttr::strip(val);
  return (isa<To>(val) || ...);
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) sugarCast(From val) {
  static_assert(isValidSugarCastType<From>,
                "sugared casts only work with Type and Attribute");
  return cast<To>(SugarAttr::strip(val));
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) sugarDynCast(From val) {
  static_assert(isValidSugarCastType<From>,
                "sugared casts only work with Type and Attribute");
  return dyn_cast<To>(SugarAttr::strip(val));
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) sugarDynCastIfPresent(From val) {
  if (!val)
    return To();
  return sugarDynCast<To>(val);
}

} // namespace M::KGEN

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

/// Unwrap a type reference to get to the underlying TypeGeneratorRefAttr or
/// TypeInstanceRefAttr. Types passed through generic parameters are wrapped in
/// TypeParamAttr, and this helper handles that unwrapping.
/// Returns a null TypedAttr if the type reference cannot be resolved.
TypedAttr getTypeRefForTypeValueIfResolved(TypedAttr typeRef);
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENATTRS_H
