//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPATTRS_TD
#define KGEN_POPDIALECT_POPATTRS_TD

include "KGEN/KGENDialect/KGENAttrInterfaces.td"
include "KGEN/POPDialect/POPDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SubElementInterfaces.td"

class POP_Enum<string name, string summary, list<I32EnumAttrCase> cases> :
  I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::M::KGEN::POP";
  let genSpecializedAttr = 0;
}

class POP_EnumCase<string sym, int val, string str = sym> :
  I32EnumAttrCase<sym, val, str>;

class POP_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<POPDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// CmpPredicate
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// PrefetchTag
//===----------------------------------------------------------------------===//

def POP_PrefetchTag : POP_Enum<
    "PrefetchTag",
    "indicate pop.prefetch purpose and cache destination", [
      POP_EnumCase<"ReadICache", 0>,
      POP_EnumCase<"ReadDCache", 1>,
      POP_EnumCase<"WriteDCache", 2>
    ]>;

def POP_PrefetchTagAttr : EnumAttr<POPDialect, POP_PrefetchTag, "prefetch_tag">;

//===----------------------------------------------------------------------===//
// PrefetchLocality
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// SIMDAttr
//===----------------------------------------------------------------------===//

def POP_SIMDAttr : POP_Attr<"SIMD", "simd", [TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "A SIMD vector attribute.";
  let description = [{
    The `#pop.simd` attribute represents a constant SIMD vector value. It
    contains `N` values of a particular dtype. Only integer, floating point, and
    bool dtypes are supported.

    Example:

    ```mlir
    #pop.simd<1, 2> : !pop.simd<2, si32>
    #pop.simd<1.5, 2.5> : !pop.simd<2, f64>
    #pop.simd<true, false> : !pop.simd<2, bool>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"DTypeValue">:$values,
                        AttributeSelfTypeParameter<"", "SIMDType">:$type);
  let assemblyFormat = [{
    `<` (`>`) : (custom<DTypeValues>($values, ref($type))^ `>`)?
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

def POP_ArrayAttr : POP_Attr<"Array", "array", [TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "An array attribute.";
  let description = [{
    The `#pop.array` attribute contains a homogeneous list of elements of an
    array type. It is used to represent array constant values.

    Example:

    ```mlir
    #pop.array<1, 2> : !pop.array<2, index>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "ArrayType">:$type);

  let assemblyFormat = [{
    `<` (`>`) : (custom<ArrayElements>($values, ref($type))^ `>`)?
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// StructAttr
//===----------------------------------------------------------------------===//

def POP_StructAttr : POP_Attr<"Struct", "struct", [TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "A heterogeneous list of elements.";
  let description = [{
    The `#pop.struct` attribute contains a heterogenous list of elements of
    struct type. It is used to represent constant struct values.

    Example:

    ```mlir
    #pop.struct<3, 3.5> : !pop.struct<index, f32>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "StructType">:$type);

  let assemblyFormat = [{
    `<` (`>`) : (custom<StructElements>($values, ref($type))^ `>`)?
  }];

  let genVerifyDecl = 1;
}

#endif // KGEN_POPDIALECT_POPATTRS_TD
