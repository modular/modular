//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef INDEXENUMS_TD
#define INDEXENUMS_TD

include "Support/IndexDialect/IndexDialect.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// IndexCmpPredicate
//===----------------------------------------------------------------------===//

def IndexCmpPredicate : I32EnumAttr<
    "IndexCmpPredicate", "index comparison predicate kind",
    [
      I32EnumAttrCase<"EQ", 0, "eq">,
      I32EnumAttrCase<"NE", 1, "ne">,
      I32EnumAttrCase<"SLT", 2, "slt">,
      I32EnumAttrCase<"SLE", 3, "sle">,
      I32EnumAttrCase<"SGT", 4, "sgt">,
      I32EnumAttrCase<"SGE", 5, "sge">,
      I32EnumAttrCase<"ULT", 6, "ult">,
      I32EnumAttrCase<"ULE", 7, "ule">,
      I32EnumAttrCase<"UGT", 8, "ugt">,
      I32EnumAttrCase<"UGE", 9, "uge">
    ]> {
  let cppNamespace = "::M::index";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// IndexCmpPredicateAttr
//===----------------------------------------------------------------------===//

def IndexCmpPredicateAttr : EnumAttr<
    IndexDialect, IndexCmpPredicate, "cmp_predicate">;

#endif // INDEXENUMS_TD
