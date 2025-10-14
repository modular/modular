//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITATTRS_H
#define KGEN_LITDIALECT_LITATTRS_H

#include "KGEN/KGENDialect/KGENAttrInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Regex.h"

namespace M::KGEN {
class NoneType;
namespace LIT {
class ModuleType;
class StructMetaType;
class StructType;
class OriginType;
class OriginSetType;
class RefPackType;
class TraitType;
class StructFieldOp;
class FnMetadataAttr;
} // namespace LIT
} // namespace M::KGEN

#include "KGEN/LITDialect/LITEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "KGEN/LITDialect/LITAttrs.h.inc"

namespace M::KGEN::LIT {

/// Given an attribute or type, return the "canonical" version of the attribute
/// with all type sugar removed.
TypedAttr getCanonicalAttr(TypedAttr src);
Type getCanonicalType(Type type);

/// Given a list of operations, create an array of bools (as a mask) indicating
/// variadic parameters in their concatenated list of parameter declarations.
/// The given operations must all implement DeclInterface.
SmallVector<VariadicKind>
getContextualVariadicParams(ArrayRef<Operation *> ops);

/// This digs in and unpacks all of the origin references in the specified
/// TypedAttr unpacking unions, but maintaining mutability.  This typically
/// will return ParamRefAttr's or ImmutCast(ParamRefAttr)'s if a mutable
/// origin is accessed immutably.
///
/// This invokes the specified closure on each origin element.
template <typename T>
static inline void processRawOrigin(TypedAttr origin, T &&fn) {
  assert(!isa<SugarAttr>(origin) && "origins should be canonicalized");

  // Expand origin unions into their members, we know they will canonicalize
  // nested unions into a single one.
  if (auto unionAttr =
          dyn_cast<OriginUnionAttr>(OriginMutCastAttr::strip(origin))) {
    // If we stripped a MutCastAttr off the outer union, put it onto each
    // element we return.
    bool needsImmutCast = TypedAttr(unionAttr) != origin;
    for (auto elt : unionAttr.getOperands()) {
      if (needsImmutCast)
        elt = OriginMutCastAttr::get(elt, origin.getType());
      fn(elt);
    }
    return;
  }

  fn(origin);
}

// Helpers for sugar-aware casting.
template <typename... To, typename From>
[[nodiscard]] inline bool sugarIsa(From val) {
  val = SugarAttr::strip(val);
  return (isa<To>(val) || ...);
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) sugarCast(From val) {
  val = SugarAttr::strip(val);
  return cast<To>(val);
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) sugarDynCast(From val) {
  val = SugarAttr::strip(val);
  return dyn_cast<To>(val);
}

} // namespace M::KGEN::LIT

#endif // KGEN_LITDIALECT_LITATTRS_H
