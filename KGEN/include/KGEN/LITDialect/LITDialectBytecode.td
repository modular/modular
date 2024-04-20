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

def UnpackedType : DialectType<(type)>;

def DeclRefType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  Type:$metaType
)>;

def AnyStructType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  ConcreteType<"TypeSignatureType">:$signature
)>;

def TraitType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$symbol
)>;

def AnyTraitType : DialectType<(type
  ConcreteType<"TraitType">:$traitType
)>;

def LifetimeType : DialectType<(type
  TypedAttr:$isMutable
)>;

def RefType : DialectType<(type
  Type:$elementType,
  TypedAttr:$lifetime,
  TypedAttr:$addressSpace
)>;

def RefPackType : DialectType<(type
  TypedAttr:$variadic,
  TypedAttr:$lifetime,
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
    UnpackedType,
    DeclRefType,
    AnyStructType,
    TraitType,
    LifetimeType,
    RefType,
    REPLResultRefType,
    MagicMLIRAttrType,
    MagicMLIROpType,
    MagicMLIRTypeType,
    RefPackType,
    AnyTraitType
  ];
}

//===----------------------------------------------------------------------===//
// LITDialect Attributes
//===----------------------------------------------------------------------===//

def PogMetadataAttr : DialectAttribute<(attr
  StringAttr:$name,
  EnumAttr<"PassingKind">:$passingKind,
  Bool:$isVariadic
)>;

def PogListAttr : DialectAttribute<(attr
  ConcreteAttrArray<"PogMetadataAttr">:$pogs,
  TypedAttrArray:$defaultPos,
  TypedAttrArray:$defaultKwOnly,
  VarInt:$packIndex,
  EnumAttr<"ArgConvention">:$origPackConvention
)>;

def FnMetadataAttr : DialectAttribute<(attr
  ConcreteAttribute<"PogListAttr">:$argListAttrs,
  ConcreteAttribute<"PogListAttr">:$paramListAttrs,
  VarInt:$numImplicitLifetimeDecls
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
  ConcreteType<"UnpackedType">:$type
)>;

def ModuleAttr : DialectAttribute<(attr
  ConcreteType<"AnyStructType">:$type
)>;

def BindTypeAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  TypedAttrArray:$values,
  ConcreteType<"AnyStructType">:$type
)>;

def LifetimeAttr : DialectAttribute<(attr
  ConcreteType<"LifetimeType">:$type
)>;

def LifetimeUnionAttr : DialectAttribute<(attr
  TypedAttrArray:$operands,
  ConcreteType<"LifetimeType">:$type
)>;

def LifetimeMutCastAttr : DialectAttribute<(attr
  TypedAttr:$operand,
  ConcreteType<"LifetimeType">:$type
)>;

def ImplicitLifetimeRefAttr : DialectAttribute<(attr
  VarInt:$depth,
  VarInt:$index,
  ConcreteType<"LifetimeType">:$type
)>;

def UnresolvedAliasValueAttr : DialectAttribute<(attr
  Type:$type
)>;

def LITStructAttr : DialectAttribute<(attr
  WithParser<"succeeded(readStructValues($_reader, $_var))",
  WithPrinter<"writeStructValues($_writer, $_getter)",
  WithType<"SmallVector<std::tuple<StringAttr, TypedAttr>>">>>:$values,
  ConcreteType<"DeclRefType">:$type
)>;

def StructExtractAttr : DialectAttribute<(attr
  TypedAttr:$structValue,
  StringAttr:$field,
  Type:$type
)>;

def TypeLineageAttr : DialectAttribute<(attr
  Type:$type,
  TypeArray:$inheritedFrom
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

def InvalidRefLifetimeAttr : DialectAttribute<(attr
  ConcreteType<"LifetimeType">:$type
)>;

def RefPackAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"RefPackType">:$type
)>;

def TypeLineageArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"TypeLineageAttr">:$value
)>;

def LITDialectAttributes : DialectAttributes<"lit"> {
  let elems = [
    PogMetadataAttr,
    PogListAttr,
    FnMetadataAttr,
    DocStringAttr,
    UnboundMLIROperationAttr,
    UnpackedAttr,
    ModuleAttr,
    BindTypeAttr,
    LifetimeAttr,
    ImplicitLifetimeRefAttr,
    UnresolvedAliasValueAttr,
    LITStructAttr,
    StructExtractAttr,
    TypeLineageAttr,
    TypeConventionAttr,
    LifetimeUnionAttr,
    LifetimeMutCastAttr,
    VarDeclKindAttr,
    InvalidRefLifetimeAttr,
    RefPackAttr,
    TypeLineageArrayAttr
  ];
}

#endif // LITDIALECT_BYTECODE
