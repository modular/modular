//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPDIALECTBYTECODE_TD
#define KGEN_POPDIALECT_POPDIALECTBYTECODE_TD

include "Support/Compiler/Bytecode.td"

//===----------------------------------------------------------------------===//
// POPDialect Types
//===----------------------------------------------------------------------===//

def ArrayType : DialectType<(type
  TypedAttr:$size,
  Type:$elementType
)>;

def UnionType : DialectType<(type
  ConcreteTypeArray<"Type">:$types
)>;

def SIMDType : DialectType<(type
  TypedAttr:$size,
  TypedAttr:$dType
)>;

def IntLiteralType : DialectType<(type)>;

def FloatLiteralType : DialectType<(type)>;

def POPDialectTypes : DialectTypes<"pop"> {
  let elems = [
    ArrayType,
    UnionType,
    SIMDType,
    IntLiteralType,
    FloatLiteralType
  ];
}

//===----------------------------------------------------------------------===//
// POPDialect Attributes
//===----------------------------------------------------------------------===//

def CmpPredicateAttr : DialectAttribute<(attr
  EnumAttr<"CmpPredicate">:$value
)>;

def NormalizedCmpPredicateAttr : DialectAttribute<(attr
  EnumAttr<"NormalizedCmpPredicate">:$value
)>;

def AtomicOrderingAttr : DialectAttribute<(attr
  EnumAttr<"AtomicOrdering">:$value
)>;

def AtomicBinOpAttr : DialectAttribute<(attr
  EnumAttr<"AtomicBinOp">:$value
)>;

def FastmathFlagsAttr : DialectAttribute<(attr
  EnumAttr<"FastmathFlags">:$value
)>;

def UnionAttr : DialectAttribute<(attr
  TypedAttr:$value,
  ConcreteType<"UnionType">:$type
)>;

def DTypeValues :
  WithParser<"succeeded(readDTypeValues($_reader, $_var))",
  WithPrinter<"writeDTypeValues($_writer, $_getter)",
  WithType<"SmallVector<DTypeValue>">>>;

def SIMDAttr : DialectAttribute<(attr
  DTypeValues:$values,
  ConcreteType<"SIMDType">:$type
)>;

def ArrayAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"ArrayType">:$type
)>;

def IntLiteralAttr : DialectAttribute<(attr
  Custom<"IPInt">:$value
)>;

def IntLiteralConvertAttr : DialectAttribute<(attr
  Type:$type,
  TypedAttr:$input,
  ConcreteAttribute<"UnitAttr">:$treatIndexAsUnsigned
)>;

def IntLiteralBinOpAttr : DialectAttribute<(attr
  Type:$type,
  TypedAttr:$lhs, TypedAttr:$rhs,
  ConcreteAttribute<"IntLiteralBinKindAttr">:$kind
)>;


def IntLiteralCmpPredAttr : DialectAttribute<(attr
  EnumAttr<"IntLiteralCmpPred">:$value
)>;

def IntLiteralBinKindAttr : DialectAttribute<(attr
  EnumAttr<"IntLiteralBinKind">:$value
)>;

def FloatLiteralAttr : DialectAttribute<(attr
  ConcreteAttribute<"FloatLiteralSpecialValuesAttr">:$special,
  Custom<"OptionalIPRational">:$rational
)>;

def FloatLiteralSpecialValuesAttr : DialectAttribute<(attr
  EnumAttr<"FloatLiteralSpecialValues">:$value
)>;

def FloatLiteralBinKindAttr : DialectAttribute<(attr
  EnumAttr<"FloatLiteralBinKind">:$value
)>;

def FloatLiteralCmpPredAttr : DialectAttribute<(attr
  EnumAttr<"FloatLiteralCmpPred">:$value
)>;

def SIMDAndAttr : DialectAttribute<(attr TypedAttr:$lhs, TypedAttr:$rhs)>;
def SIMDXorAttr : DialectAttribute<(attr TypedAttr:$lhs, TypedAttr:$rhs)>;

def SIMDCmpAttr : DialectAttribute<(attr
  EnumAttr<"NormalizedCmpPredicate">:$cc, TypedAttr:$lhs, TypedAttr:$rhs, Type:$type)>;

def CastAttr : DialectAttribute<(attr
  TypedAttr:$arg, Type:$type
)>;

def DTypeToUI8Attr : DialectAttribute<(attr TypedAttr:$value)>;
def DTypeFromUI8Attr : DialectAttribute<(attr TypedAttr:$value)>;

def CastFromBuiltinAttr : DialectAttribute<(attr
  TypedAttr:$arg, ConcreteType<"SIMDType">:$type
)>;
def CastToBuiltinAttr : DialectAttribute<(attr
  TypedAttr:$arg, Type:$type
)>;

def SIMDSplatAttr : DialectAttribute<(attr
  TypedAttr:$arg, ConcreteType<"SIMDType">:$type
)>;

def SIMDReduceOrAttr : DialectAttribute<(attr
  TypedAttr:$vector, ConcreteType<"SIMDType">:$type
)>;
def SIMDReduceAndAttr : DialectAttribute<(attr
  TypedAttr:$vector, ConcreteType<"SIMDType">:$type
)>;

def POPDialectAttributes : DialectAttributes<"pop"> {
  let elems = [
    CmpPredicateAttr,
    AtomicOrderingAttr,
    AtomicBinOpAttr,
    FastmathFlagsAttr,
    UnionAttr,
    SIMDAttr,
    ArrayAttr,

    IntLiteralCmpPredAttr,
    IntLiteralBinKindAttr,
    FloatLiteralSpecialValuesAttr,
    FloatLiteralCmpPredAttr,
    FloatLiteralBinKindAttr,

    IntLiteralAttr,
    FloatLiteralAttr,

    SIMDAndAttr,
    SIMDXorAttr,

    CastAttr,
    DTypeToUI8Attr,
    DTypeFromUI8Attr,
    CastFromBuiltinAttr,
    CastToBuiltinAttr,
    SIMDSplatAttr,

    SIMDCmpAttr,
    SIMDReduceOrAttr,
    SIMDReduceAndAttr,
  ];
}

#endif // KGEN_POPDIALECT_POPDIALECTBYTECODE_TD
