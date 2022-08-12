//===- POPAttrs.td --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPATTRS_TD
#define KGEN_POPDIALECT_POPATTRS_TD

include "KGEN/POPDialect/POPDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SubElementInterfaces.td"

def CmpPredicate : I32EnumAttr<
    "CmpPredicate", "pop.cmp comparison predicate", [
      I32EnumAttrCase<"EQ", 0, "eq">,
      I32EnumAttrCase<"NE", 1, "ne">,
      I32EnumAttrCase<"LT", 2, "lt">,
      I32EnumAttrCase<"GT", 3, "gt">,
      I32EnumAttrCase<"LE", 4, "le">,
      I32EnumAttrCase<"GE", 5, "ge">
    ]> {
  let cppNamespace = "::M::KGEN";
  let genSpecializedAttr = 0;
}

def CmpPredicateAttr : EnumAttr<POPDialect, CmpPredicate, "cmp_pred">;

#endif // KGEN_POPDIALECT_POPATTRS_TD
