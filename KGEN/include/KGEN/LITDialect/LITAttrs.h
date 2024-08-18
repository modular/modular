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
class AnyStructType;
class StructType;
class LifetimeType;
class LifetimeSetType;
class RefPackType;
class TraitType;
class UnpackedType;
class StructFieldOp;
class FnMetadataAttr;
} // namespace LIT
} // namespace M::KGEN

#include "KGEN/LITDialect/LITEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "KGEN/LITDialect/LITAttrs.h.inc"

namespace M::KGEN::LIT {

/// Given a list of operations, create an array of bools (as a mask) indicating
/// variadic parameters in their concatenated list of parameter declarations.
/// The given operations must all implement DeclInterface.
SmallVector<bool> getContextualVariadicMask(ArrayRef<Operation *> ops);

/// This digs in and unpacks all of the lifetime references in the specified
/// TypedAttr unpacking unions, but maintaining mutability.  This typically
/// will return ParamRefAttr's or ImmutCast(ParamRefAttr)'s if a mutable
/// lifetime is accessed immutably.
///
/// This invokes the specified closure on each lifetime element.
template <typename T>
static inline void processRawLifetime(TypedAttr lifetime, T &&fn) {
  // Expand lifetime unions into their members, we know they will canonicalize
  // nested unions into a single one.
  if (auto unionAttr =
          dyn_cast<LifetimeUnionAttr>(LifetimeMutCastAttr::strip(lifetime))) {
    // If we stripped a MutCastAttr off the outer union, put it onto each
    // element we return.
    bool needsImmutCast = TypedAttr(unionAttr) != lifetime;
    for (auto elt : unionAttr.getOperands()) {
      if (needsImmutCast)
        elt = LifetimeMutCastAttr::get(elt, lifetime.getType());
      fn(elt);
    }
    return;
  }

  fn(lifetime);
}

} // namespace M::KGEN::LIT

#endif // KGEN_LITDIALECT_LITATTRS_H
