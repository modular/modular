//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_ATTRBASE_TD
#define SUPPORT_COMPILER_ATTRBASE_TD

include "mlir/IR/BuiltinAttributeInterfaces.td"

class AttrOfType<Type type> : ConfinedAttr<TypedAttrInterface, [
  AttrConstraint<SubstLeaves<"$_self", "$_self.cast<TypedAttr>().getType()",
                             type.predicate>, type.summary>]>;

#endif // SUPPORT_COMPILER_ATTRBASE_TD
