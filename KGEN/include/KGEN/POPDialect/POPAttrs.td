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

class POP_Enum<string name, string summary, list<I32EnumAttrCase> cases> : 
  I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::M::KGEN::POP";
  let genSpecializedAttr = 0;
}

class POP_EnumCase<string sym, int val, string str = sym> : 
  I32EnumAttrCase<sym, val, str>;

def POP_CmpPredicate : POP_Enum<
    "CmpPredicate", "pop.cmp comparison predicate", [
      POP_EnumCase<"EQ", 0, "eq">,
      POP_EnumCase<"NE", 1, "ne">,
      POP_EnumCase<"LT", 2, "lt">,
      POP_EnumCase<"GT", 3, "gt">,
      POP_EnumCase<"LE", 4, "le">,
      POP_EnumCase<"GE", 5, "ge">
    ]>;

def POP_CmpPredicateAttr : EnumAttr<POPDialect, POP_CmpPredicate, "cmp_pred">;

def POP_PrefetchTag : POP_Enum<
    "PrefetchTag", 
    "indicate pop.prefetch purpose and cache destination", [
      POP_EnumCase<"ReadICache", 0>,
      POP_EnumCase<"ReadDCache", 1>,
      POP_EnumCase<"WriteDCache", 2>
    ]>;

def POP_PrefetchTagAttr : EnumAttr<POPDialect, POP_PrefetchTag, "prefetch_tag">;

def POP_PrefetchLocality : POP_Enum<
    "PrefetchLocality", 
    "indicate temporal locality of prefetched data ", [
      POP_EnumCase<"NoLocality", 0>,
      POP_EnumCase<"LowLocality", 1>,
      POP_EnumCase<"MediumLocality", 2>,
      POP_EnumCase<"HighLocality", 3>,
      POP_EnumCase<"VeryHighLocality", 4>
    ]>;

def POP_PrefetchLocalityAttr : 
  EnumAttr<POPDialect, POP_PrefetchLocality, "prefetch_tag">{
    // Remove the leading space so we can define format like 
    //  `(NoLocality, ReadDCache)` in prefetch op.
    let assemblyFormat = "`` $value";
}

#endif // KGEN_POPDIALECT_POPATTRS_TD
