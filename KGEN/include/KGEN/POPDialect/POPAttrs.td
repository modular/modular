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
// AtomicOrdering
//===----------------------------------------------------------------------===//

def POP_AtomicOrdering : POP_Enum<
    "AtomicOrdering", "pop.atomic_cmp_xchg atomic ordering", [
      POP_EnumCase<"NOT_ATOMIC", 0, "not_atomic">,
      POP_EnumCase<"UNORDERED", 1, "unordered">,
      POP_EnumCase<"MONOTONIC", 2, "monotonic">,
      POP_EnumCase<"ACQUIRE", 3, "acquire">,
      POP_EnumCase<"RELEASE", 4, "release">,
      POP_EnumCase<"ACQUIRE_RELEASE", 5, "acq_rel">,
      POP_EnumCase<"SEQUENTIALLY_CONSISTENT", 6, "seq_cst">
    ]>;

def POP_AtomicOrderingAttr : EnumAttr<POPDialect,
                                      POP_AtomicOrdering,
                                      "atomic_ordering">;

//===----------------------------------------------------------------------===//
// AtomicBinOp
//===----------------------------------------------------------------------===//

def POP_AtomicBinOp : POP_Enum<
    "AtomicBinOp", "pop.atomic.rmw operator", [
      POP_EnumCase<"XCHG", 0, "xchg">,
      POP_EnumCase<"ADD", 1, "add">,
      POP_EnumCase<"SUB", 2, "sub">,
      POP_EnumCase<"AND", 3, "and">,
      POP_EnumCase<"NAND", 4, "nand">,
      POP_EnumCase<"OR", 5, "or">,
      POP_EnumCase<"XOR", 6, "xor">,
      POP_EnumCase<"MAX", 7, "max">,
      POP_EnumCase<"MIN", 8, "min">
    ]>;

def POP_AtomicBinOpAttr : EnumAttr<POPDialect,
                                   POP_AtomicBinOp,
                                   "bin_op">;

//===----------------------------------------------------------------------===//
// FastmathFlags
//===----------------------------------------------------------------------===//

def POP_FastmathFlags : I32BitEnumAttr<"FastmathFlags", "fastmath flags", [
      I32BitEnumAttrCaseNone<"NONE", "none">,
      I32BitEnumAttrCaseBit<"NNAN", 0, "nnan">,
      I32BitEnumAttrCaseBit<"NINF", 1, "ninf">,
      I32BitEnumAttrCaseBit<"NSZ", 2, "nsz">,
      I32BitEnumAttrCaseBit<"ARCP", 3, "arcp">,
      I32BitEnumAttrCaseBit<"CONTRACT", 4, "contract">,
      I32BitEnumAttrCaseBit<"AFN", 5, "afn">,
      I32BitEnumAttrCaseBit<"REASSOC", 6, "reassoc">
    ]> {
  let cppNamespace = "::M::KGEN::POP";
  let genSpecializedAttr = 0;
}

def POP_FastmathFlagsAttr : EnumAttr<POPDialect, POP_FastmathFlags, "fmf">;

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

    When all values of the SIMD vector are equal, the attribute has a special
    splat syntax:

    ```mlir
    #pop<simd 0> : !pop.simd<4, si32>
    #pop<simd "1.5"> : !pop.simd<4, f32>
    #pop<simd false> : !pop.simd<4, bool>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"DTypeValue">:$values,
                        AttributeSelfTypeParameter<"", "SIMDType">:$type);
  let assemblyFormat = "`` custom<SIMDValues>($values, ref($type))";

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<DTypeValue>":$values,
                                        "SIMDType":$type), [{
      return get(type.getContext(), values, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "DTypeValue":$value,
                                        "SIMDType":$type), [{
      return get(ArrayRef(value), type);
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

def POP_ArrayAttr : POP_Attr<"Array", "array", [
    TypedAttrInterface,
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
    `<` (`>`) : (custom<SequenceElements>($values, ref($type))^ `>`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "ArrayType":$type), [{
      return get(type.getContext(), values, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// StructAttr
//===----------------------------------------------------------------------===//

def POP_StructAttr : POP_Attr<"Struct", "struct", [
    TypedAttrInterface,
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

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "StructType":$type), [{
      return get(type.getContext(), values, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// StructExtractAttr
//===----------------------------------------------------------------------===//

def POP_StructExtractAttr : POP_Attr<"StructExtract", "struct.extract",
    [TypedAttrInterface]> {
  let summary = "Extract an element from a struct value";
  let description = [{
    The `#pop.struct.extract` attribute represents a field reference from a
    constant struct, which may be parametric.

    Example:

    ```mlir
    #pop.struct.extract<p, 4> : index
    ```
  }];

  let parameters = (ins "TypedAttr":$structValue, "int64_t":$fieldNo,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat =
    "`<` custom<ColonTypeParamValue>($structValue) `,` $fieldNo `>`";

  let skipDefaultBuilders = 1;

  let builders = [
    // Override the default builder used by the generated parser.  This computes
    // the result type correctly and performs inline folding.
    AttrBuilderWithInferredContext<
                (ins "TypedAttr":$structValue, "int64_t":$fieldNo),
                     "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$structValue, "int64_t":$fieldNo,
                     "Type":$resultType), "", "TypedAttr">

  ];
}

//===----------------------------------------------------------------------===//
// PackAttr
//===----------------------------------------------------------------------===//

def POP_PackAttr : POP_Attr<"Pack", "pack", [
    TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "heterogeneous list of constant elements";
  let description = [{
    The `#pop.pack` attribute contains a heterogenously typed list of constant
    elements. It can be used to represent constant pack values, and so is of
    pack type.

    Example:

    ```mlir
    // A pack of 3 elements.
    %0 = kgen.param.constant: !pop.pack<[i8, ui4, i32]> = <<3, 1, 4>>
    // An empty pack.
    %1 = kgen.param.constant: !pop.pack<[]> = <<>>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "PackType">:$type);

  let assemblyFormat = [{
    `<` (`>`) : (custom<PackElements>($values, ref($type))^ `>`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "PackType":$type), [{
      return get(type.getContext(), values, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// VariantAttr
//===----------------------------------------------------------------------===//

def POP_VariantAttr : POP_Attr<"Variant", "variant", [
    TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "A variant constant value.";
  let description = [{
    The `#pop.variant` attribute represents a constant `!pop.variant` value.
    It contains a single typed attribute where the typed of the value is
    expected to equal one of the corresponding variant types.

    Example:

    ```mlir
    #pop.variant<3 : index> : !pop.variant<index, i32, f32>
    ```
  }];

  let parameters = (ins "TypedAttr":$value,
                        AttributeSelfTypeParameter<"", "VariantType">:$type);

  let assemblyFormat = "`<` custom<ColonTypeParamValue>($value) `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value,
                                        "VariantType":$type), [{
      return get(type.getContext(), value, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

#endif // KGEN_POPDIALECT_POPATTRS_TD
