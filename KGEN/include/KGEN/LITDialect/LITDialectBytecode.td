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

def StructType : DialectType<(type
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

def LifetimeSetType : DialectType<(type)>;

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
    StructType,
    AnyStructType,
    TraitType,
    AnyTraitType,
    LifetimeType,
    LifetimeSetType,
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
  Bool:$isVariadic
)>;

def OptionalArgConvention :
  WithParser<"succeeded(readOptional($_reader, $_var, &readIntegral<ArgConvention>))",
  WithPrinter<"writeOptional($_writer, $_getter, &writeIntegral<ArgConvention>)",
  WithType<"std::optional<ArgConvention>">>>;

def PogListAttr : DialectAttribute<(attr
  ConcreteAttrArray<"PogMetadataAttr">:$pogs,
  TypedAttrArray:$defaultPos,
  TypedAttrArray:$defaultKwOnly,
  VarInt:$packIndex,
  OptionalArgConvention:$origPackConvention
)>;

def FnMetadataAttr : DialectAttribute<(attr
  ConcreteAttribute<"PogListAttr">:$argListAttrs,
  ConcreteAttribute<"PogListAttr">:$paramListAttrs,
  VarInt:$numImplicitLifetimeDecls,
  TypedAttr:$captureLifetimes
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
  Bool:$kwOnly
)>;

def ModuleAttr : DialectAttribute<(attr
  ConcreteType<"AnyStructType">:$type
)>;

def WrappedBindTypeAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  TypedAttrArray:$values,
  ConcreteType<"AnyStructType">:$type
)>;

def AnyLifetimeAttr : DialectAttribute<(attr
  ConcreteType<"LifetimeType">:$type
)>;

def WrappedLifetimeUnionAttr : DialectAttribute<(attr
  TypedAttrArray:$operands,
  ConcreteType<"LifetimeType">:$type
)>;

def WrappedLifetimeMutCastAttr : DialectAttribute<(attr
  TypedAttr:$operand,
  ConcreteType<"LifetimeType">:$type
)>;

def WrappedLifetimeFieldAttr : DialectAttribute<(attr
  TypedAttr:$structLifetime,
  StringAttr:$field,
  ConcreteType<"LifetimeType">:$type
)>;

def ImplicitLifetimeRefAttr : DialectAttribute<(attr
  VarInt:$depth,
  VarInt:$index,
  ConcreteType<"LifetimeType">:$type
)>;

def WrappedLifetimeSetAttr : DialectAttribute<(attr
  TypedAttrArray:$operands,
  ConcreteType<"LifetimeSetType">:$type
)>;

def WrappedLifetimeSetUnionAttr : DialectAttribute<(attr
  TypedAttr:$value,
  ConcreteType<"LifetimeType">:$type
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

def StructGERAttr : DialectAttribute<(attr
  TypedAttr:$value,
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

def RefPackAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"RefPackType">:$type
)>;

def TypeLineageArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"TypeLineageAttr">:$value
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
    WrappedBindTypeAttr,
    AnyLifetimeAttr,
    WrappedLifetimeUnionAttr,
    WrappedLifetimeMutCastAttr,
    WrappedLifetimeFieldAttr,
    ImplicitLifetimeRefAttr,
    WrappedLifetimeSetAttr,
    WrappedLifetimeSetUnionAttr,
    UnresolvedAliasValueAttr,
    LITStructAttr,
    WrappedStructExtractAttr,
    StructGERAttr,
    TypeLineageAttr,
    TypeLineageArrayAttr,
    RefPackAttr,
    TypeConventionAttr,
    VarDeclKindAttr,
  ];
}

#endif // LITDIALECT_BYTECODE
