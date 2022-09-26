//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPATTRS_TD
#define KGEN_POPDIALECT_POPATTRS_TD

include "KGEN/POPDialect/POPDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SubElementInterfaces.td"

def POP_CmpPredicate : I32EnumAttr<
    "CmpPredicate", "pop.cmp comparison predicate", [
      I32EnumAttrCase<"EQ", 0, "eq">,
      I32EnumAttrCase<"NE", 1, "ne">,
      I32EnumAttrCase<"LT", 2, "lt">,
      I32EnumAttrCase<"GT", 3, "gt">,
      I32EnumAttrCase<"LE", 4, "le">,
      I32EnumAttrCase<"GE", 5, "ge">
    ]> {
  let cppNamespace = "::M::KGEN::POP";
  let genSpecializedAttr = 0;
}

def POP_CmpPredicateAttr : EnumAttr<POPDialect, POP_CmpPredicate, "cmp_pred">;

#endif // KGEN_POPDIALECT_POPATTRS_TD
