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


def FASTMATH_NONE            : I32BitEnumAttrCaseNone<"none"      >;
def FASTMATH_NO_NANS         : I32BitEnumAttrCaseBit<"nnan",     0>;
def FASTMATH_NO_INFS         : I32BitEnumAttrCaseBit<"ninf",     1>;
def FASTMATH_NO_SIGNED_ZEROS : I32BitEnumAttrCaseBit<"nsz",      2>;
def FASTMATH_ALLOW_RECIP     : I32BitEnumAttrCaseBit<"arcp",     3>;
def FASTMATH_ALLOW_CONTRACT  : I32BitEnumAttrCaseBit<"contract", 4>;
def FASTMATH_APPROX_FUNC     : I32BitEnumAttrCaseBit<"afn",      5>;
def FASTMATH_REASSOC         : I32BitEnumAttrCaseBit<"reassoc",  6>;
def FASTMATH_FAST            : I32BitEnumAttrCaseGroup<
    "fast",
    [
      FASTMATH_NO_NANS,
      FASTMATH_NO_INFS,
      FASTMATH_NO_SIGNED_ZEROS,
      FASTMATH_ALLOW_RECIP,
      FASTMATH_ALLOW_CONTRACT,
      FASTMATH_APPROX_FUNC,
      FASTMATH_REASSOC
    ]>;

def POP_FastmathFlags : I32BitEnumAttr<"FastmathFlags", "fastmath flags", [
      FASTMATH_NONE,
      FASTMATH_NO_NANS,
      FASTMATH_NO_INFS,
      FASTMATH_NO_SIGNED_ZEROS,
      FASTMATH_ALLOW_RECIP,
      FASTMATH_ALLOW_CONTRACT,
      FASTMATH_APPROX_FUNC,
      FASTMATH_REASSOC,
      FASTMATH_FAST
]> {
  let cppNamespace = "::M::KGEN::POP";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def POP_FastmathFlagsAttr : EnumAttr<POPDialect, POP_FastmathFlags, "fmf">;

//===----------------------------------------------------------------------===//
// UnionAttr
//===----------------------------------------------------------------------===//

def POP_UnionAttr : POP_Attr<"Union", "union", [TypedAttrInterface]> {
  let summary = "a union with a specific typed value";
  let description = [{
    The `#kgen.union` attribute represents a union with a value of one of the
    union's element types. The attribute knows which union element the value is
    and contains an attribute of said value type.

    Example:

    ```mlir
    #kgen.union<0> : !kgen.union<index, struct<()>>
    #kgen.union<:i64 42> : !kgen.union<i64, none>
    ```
  }];

  let parameters = (ins "TypedAttr":$value,
                         AttributeSelfTypeParameter<"", "UnionType">:$type);
  let assemblyFormat = "`<` custom<ColonTypeParamValue>($value) `>`";
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value, "UnionType":$type), [{
      return $_get(type.getContext(), value, type);
    }]>
  ];
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
    }]>,
    AttrBuilderWithInferredContext<(ins "uint64_t":$intVal, "SIMDType":$type)>
  ];

  let genVerifyDecl = 1;

  /// Note: DTypeValue is not POD since APInt will allocate if its data exceeds
  /// 64 bits. Use a custom storage class to own the memory.
  let genStorageClass = 0;
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

#endif // KGEN_POPDIALECT_POPATTRS_TD
