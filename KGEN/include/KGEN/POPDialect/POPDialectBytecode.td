//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef POPDIALECT_BYTECODE
#define POPDIALECT_BYTECODE

include "Support/Compiler/Bytecode.td"

//===----------------------------------------------------------------------===//
// POPDialect Types
//===----------------------------------------------------------------------===//

def ArrayType : DialectType<(type
  TypedAttr:$size,
  TypedAttr:$elementType
)>;

def PointerType : DialectType<(type
  TypedAttr:$elementType,
  TypedAttr:$addressSpace
)>;

def SIMDType : DialectType<(type
  TypedAttr:$size,
  TypedAttr:$dType
)>;

def VariantType : DialectType<(type
  TypedAttrArray:$types
)>;

def CoroutineType : DialectType<(type
  ConcreteType<"SignatureType">:$signature
)>;

def POPDialectTypes : DialectTypes<"pop"> {
  let elems = [
    ArrayType,
    PointerType,
    SIMDType,
    VariantType,
    CoroutineType
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

def VariantAttr : DialectAttribute<(attr
  TypedAttr:$value,
  VarInt:$index,
  ConcreteType<"VariantType">:$type
)>;

def POPDialectAttributes : DialectAttributes<"pop"> {
  let elems = [
    CmpPredicateAttr,
    AtomicOrderingAttr,
    AtomicBinOpAttr,
    FastmathFlagsAttr,
    SIMDAttr,
    ArrayAttr,
    VariantAttr
  ];
}

#endif // POPDIALECT_BYTECODE
