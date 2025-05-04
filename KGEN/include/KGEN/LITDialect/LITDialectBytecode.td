//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LITDIALECT_BYTECODE
#define LITDIALECT_BYTECODE

include "Support/Compiler/Bytecode.td"

//===----------------------------------------------------------------------===//
// LITDialect Types
//===----------------------------------------------------------------------===//

def TypeCheckErrorType : DialectType<(type)>;

def UnresolvedType : DialectType<(type)>;

def TypeSignatureType : DialectType<(type
  TypeArray:$paramTypes,
  ConcreteAttribute<"PogListAttr">:$paramListAttrs
)>;

def StructType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  ConcreteType<"TypeSignatureType">:$signature
)>;

def TraitType : DialectType<(type
  ConcreteAttrArray<"SymbolRefAttr">:$symbols
)>;

def AnyTraitType : DialectType<(type
  ConcreteType<"TraitType">:$traitType
)>;

def MetaType : DialectType<(type
  Type:$type
)>;

def OriginType : DialectType<(type
  TypedAttr:$isMutable
)>;

def OriginSetType : DialectType<(type)>;

def RefType : DialectType<(type
  Type:$elementType,
  TypedAttr:$origin,
  TypedAttr:$addressSpace
)>;

def RefPackType : DialectType<(type
  TypedAttr:$variadic,
  TypedAttr:$origin,
  TypedAttr:$addressSpace
)>;

def REPLResultRefType : DialectType<(type
  Type:$elementType
)>;

def MagicMLIRAttrType : DialectType<(type)>;

def MagicMLIROpType : DialectType<(type)>;

def MagicMLIRTypeType : DialectType<(type)>;

def LITDialectTypes : DialectTypes<"lit"> {
  let elems = [
    TypeCheckErrorType,
    UnresolvedType,
    TypeSignatureType,
    StructType,
    TraitType,
    AnyTraitType,
    MetaType,
    OriginType,
    OriginSetType,
    RefType,
    RefPackType,
    REPLResultRefType,
    MagicMLIRAttrType,
    MagicMLIROpType,
    MagicMLIRTypeType
  ];
}

//===----------------------------------------------------------------------===//
// LITDialect Attributes
//===----------------------------------------------------------------------===//

def SymbolAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$value
)>;

def PogMetadataAttr : DialectAttribute<(attr
  StringAttr:$name,
  EnumAttr<"PassingKind">:$passingKind,
  EnumAttr<"VariadicKind">:$variadic
)>;

def PogListAttr : DialectAttribute<(attr
  ConcreteAttrArray<"PogMetadataAttr">:$pogs,
  TypedAttrArray:$defaultPos,
  TypedAttrArray:$defaultKwOnly,
  EnumAttr<"ArgConvention">:$origPackConvention
)>;

def FnMetadataAttr : DialectAttribute<(attr
  ConcreteAttribute<"PogListAttr">:$argListAttrs,
  VarInt:$numImplicitOriginDecls,
  TypedAttr:$captureOrigins,
  Bool:$isNestedOriginExclusivityCheckingDisabled
)>;

def DocStringAttr : DialectAttribute<(attr
  StringAttr:$string,
  OptionalAttribute<"FileLineColLoc">:$location
)>;

def UnboundMLIROperationAttr : DialectAttribute<(attr
  StringAttr:$name,
  ConcreteAttribute<"DictionaryAttr">:$attrs
)>;

def UnpackedAttr : DialectAttribute<(attr
  TypedAttr:$value,
  Bool:$kwOnly,
  Type:$type
)>;

def ModuleAttr : DialectAttribute<(attr
  ConcreteType<"StructMetaType">:$type
)>;

def AnyOriginAttr : DialectAttribute<(attr
  ConcreteType<"OriginType">:$type
)>;

def WrappedOriginUnionAttr : DialectAttribute<(attr
  TypedAttrArray:$operands,
  ConcreteType<"OriginType">:$type
)>;

def WrappedOriginMutCastAttr : DialectAttribute<(attr
  TypedAttr:$operand,
  ConcreteType<"OriginType">:$type
)>;

def WrappedOriginFieldAttr : DialectAttribute<(attr
  TypedAttr:$base,
  StringAttr:$field,
  ConcreteType<"OriginType">:$type
)>;

def WrappedIndirectOriginAttr : DialectAttribute<(attr
  TypedAttr:$baseOrigin,
  ConcreteType<"OriginType">:$type
)>;

def ImplicitOriginRefAttr : DialectAttribute<(attr
  VarInt:$depth,
  VarInt:$index,
  ConcreteType<"OriginType">:$type
)>;

def WrappedOriginSetAttr : DialectAttribute<(attr
  TypedAttrArray:$operands,
  ConcreteType<"OriginSetType">:$type
)>;

def WrappedOriginSetUnionAttr : DialectAttribute<(attr
  TypedAttr:$value,
  ConcreteType<"OriginType">:$type
)>;

def UnresolvedAliasValueAttr : DialectAttribute<(attr
  Type:$type
)>;

def LITStructAttr : DialectAttribute<(attr
  WithParser<"succeeded(readStructValues($_reader, $_var))",
  WithPrinter<"writeStructValues($_writer, $_getter)",
  WithType<"SmallVector<std::tuple<StringAttr, TypedAttr>>">>>:$values,
  ConcreteType<"StructType">:$type
)>;

def WrappedStructExtractAttr : DialectAttribute<(attr
  TypedAttr:$structValue,
  StringAttr:$field,
  Type:$type
)>;

def TypeConventionAttr : DialectAttribute<(attr
  EnumAttr<"TypeConvention">:$value
)>;

def PassingKindAttr : DialectAttribute<(attr
  EnumAttr<"PassingKind">:$value
)>;

def VarDeclKindAttr : DialectAttribute<(attr
  EnumAttr<"VarDeclKind">:$value
)>;

def RefPackAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"RefPackType">:$type
)>;

def LITDialectAttributes : DialectAttributes<"lit"> {
  let elems = [
    SymbolAttr,
    PogMetadataAttr,
    PogListAttr,
    FnMetadataAttr,
    DocStringAttr,
    UnboundMLIROperationAttr,
    UnpackedAttr,
    ModuleAttr,
    AnyOriginAttr,
    WrappedOriginUnionAttr,
    WrappedOriginMutCastAttr,
    WrappedOriginFieldAttr,
    ImplicitOriginRefAttr,
    WrappedOriginSetAttr,
    WrappedOriginSetUnionAttr,
    UnresolvedAliasValueAttr,
    LITStructAttr,
    WrappedStructExtractAttr,
    RefPackAttr,
    TypeConventionAttr,
    VarDeclKindAttr,
  ];
}

#endif // LITDIALECT_BYTECODE
