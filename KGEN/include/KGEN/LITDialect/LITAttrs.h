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

/// Given a list of operations, create an array of bools (as a mask) indicating
/// variadic parameters in their concatenated list of parameter declarations.
/// The given operations must all implement DeclInterface.
SmallVector<VariadicKind>
getContextualVariadicParams(ArrayRef<Operation *> ops);

/// This digs in and unpacks all of the origin references in the specified
/// TypedAttr, unpacking unions.
///
/// This invokes the specified closure on each origin element.
template <typename T>
static inline void processOriginUnionElts(TypedAttr origin, T &&fn) {
  if (auto sugar = dyn_cast<SugarAttr>(origin))
    origin = sugar.getCanonical();

  // Expand origin unions into their members, we know they will canonicalize
  // nested unions into a single one.
  if (auto unionAttr = dyn_cast<OriginUnionAttr>(origin)) {
    for (auto elt : unionAttr.getOperands())
      fn(elt);
    return;
  }

  fn(origin);
}

} // namespace M::KGEN::LIT

#endif // KGEN_LITDIALECT_LITATTRS_H
