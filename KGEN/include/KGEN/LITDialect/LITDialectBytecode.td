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

def DiscardType : DialectType<(type)>;

def TypeSignatureType : DialectType<(type
  TypeArray:$paramTypes,
  ConcreteAttribute<"ArgParamListAttr">:$paramListAttrs,
  Bool:$paramVarArg
)>;

def UnpackedType : DialectType<(type
  Type:$exprType
)>;

def MetaTypeType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  ConcreteType<"TypeSignatureType">:$signature
)>;

def TraitType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$symbol
)>;

def LifetimeType : DialectType<(type
  TypedAttr:$isMutable
)>;

def RefType : DialectType<(type
  Type:$elementType,
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
    DiscardType,
    TypeSignatureType,
    UnpackedType,
    MetaTypeType,
    TraitType,
    LifetimeType,
    RefType,
    REPLResultRefType,
    MagicMLIRAttrType,
    MagicMLIROpType,
    MagicMLIRTypeType
  ];
}

//===----------------------------------------------------------------------===//
// LITDialect Attributes
//===----------------------------------------------------------------------===//

def ArgParamListAttr : DialectAttribute<(attr
  StringAttrArray:$names,
  EnumAttrArray<"PassingKind">:$passingKinds,
  TypedAttrArray:$defaultPos,
  TypedAttrArray:$defaultKwOnly,
  IntegralAttrArray<"size_t">:$variadicIndices,
  IntegralAttrArray<"size_t">:$packIndices
)>;

def FnMetadataAttr : DialectAttribute<(attr
  ConcreteAttribute<"ArgParamListAttr">:$argListAttrs,
  ConcreteAttribute<"ArgParamListAttr">:$paramListAttrs,
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
  TypedAttr:$value,
  ConcreteType<"UnpackedType">:$type
)>;

def ModuleAttr : DialectAttribute<(attr
  ConcreteType<"MetaTypeType">:$type
)>;

def BindTypeAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  TypedAttrArray:$values,
  ConcreteType<"MetaTypeType">:$type
)>;

def LifetimeAttr : DialectAttribute<(attr
  ConcreteType<"LifetimeType">:$type)>;

def LifetimeUnionAttr : DialectAttribute<(attr
  TypedAttrArray:$values)>;

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

def PassingKindArrayAttr : DialectAttribute<(attr
  EnumAttrArray<"PassingKind">:$value
)>;

def VarLetDeclKindAttr : DialectAttribute<(attr
  EnumAttr<"VarLetDeclKind">:$value
)>;

def LITDialectAttributes : DialectAttributes<"lit"> {
  let elems = [
    ArgParamListAttr,
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
    PassingKindArrayAttr,
    VarLetDeclKindAttr
  ];
}

#endif // LITDIALECT_BYTECODE
