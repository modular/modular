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

def POPDialectTypes : DialectTypes<"pop"> {
  let elems = [
    ArrayType,
    UnionType,
    SIMDType
  ];
}

//===----------------------------------------------------------------------===//
// POPDialect Attributes
//===----------------------------------------------------------------------===//

def CmpPredicateAttr : DialectAttribute<(attr
  EnumAttr<"CmpPredicate">:$value
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

def POPDialectAttributes : DialectAttributes<"pop"> {
  let elems = [
    CmpPredicateAttr,
    AtomicOrderingAttr,
    AtomicBinOpAttr,
    FastmathFlagsAttr,
    UnionAttr,
    SIMDAttr,
    ArrayAttr
  ];
}

#endif // KGEN_POPDIALECT_POPDIALECTBYTECODE_TD
