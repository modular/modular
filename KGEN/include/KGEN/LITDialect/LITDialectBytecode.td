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

def EllipsisType : DialectType<(type)>;

def ModuleType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$symbol
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def TypeSignatureType : DialectType<(type
  TypeArray:$paramTypes,
  ConcreteAttribute<"::M::KGEN::PogListAttr">:$paramListAttrs
)>;

def StructType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  ConcreteType<"TypeSignatureType">:$signature
)>;

def TraitType : DialectType<(type
  ConcreteAttrArray<"SymbolRefAttr">:$symbols,
  ConcreteAttrArray<"ConstraintAttr">:$constraints
)>;

def AnyTraitType : DialectType<(type
  ConcreteType<"TraitType">:$traitType
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def MetaType : DialectType<(type
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def OriginType : DialectType<(type
  TypedAttr:$isMutable
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def OriginSetType : DialectType<(type)>;

def RefType : DialectType<(type
  Type:$elementType,
  TypedAttr:$origin,
  TypedAttr:$addressSpace
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def RefPackType : DialectType<(type
  TypedAttr:$variadic,
  TypedAttr:$origin,
  TypedAttr:$addressSpace
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def REPLResultRefType : DialectType<(type
  Type:$elementType
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def MagicMLIRAttrType : DialectType<(type)>;

def MagicMLIROpType : DialectType<(type)>;

def MagicMLIRTypeType : DialectType<(type)>;

def MagicMLIRDeferredAttrType : DialectType<(type)>;

def NameLookupArgWildcardType : DialectType<(type)>;

def LITDialectTypes : DialectTypes<"lit"> {
  let elems = [
    TypeCheckErrorType,
    UnresolvedType,
    EllipsisType,
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
    MagicMLIRTypeType,
    MagicMLIRDeferredAttrType,
    NameLookupArgWildcardType
  ];
}

//===----------------------------------------------------------------------===//
// LITDialect Attributes
//===----------------------------------------------------------------------===//

def DenseResourceElementsArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"DenseResourceElementsAttr">:$value
)>;

def SymbolAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$value
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def FnMetadataAttr : DialectAttribute<(attr
  ConcreteAttribute<"::M::KGEN::PogListAttr">:$argListAttrs,
  VarInt:$numImplicitOriginDecls,
  TypedAttr:$captureOrigins,
  Bool:$isNestedOriginExclusivityCheckingDisabled
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DeprecationInfoAttr : DialectAttribute<(attr
  StringAttr:$warning,
  OptionalAttribute<"StringAttr">:$replacement
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DocStringAttr : DialectAttribute<(attr
  StringAttr:$string,
  OptionalAttribute<"FileLineColLoc">:$location
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def EllipsisAttr : DialectAttribute<(attr)>;

def UnboundMLIROperationAttr : DialectAttribute<(attr
  StringAttr:$name,
  ConcreteAttribute<"DictionaryAttr">:$attrs
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def UnpackedAttr : DialectAttribute<(attr
  TypedAttr:$value,
  Bool:$kwOnly,
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def ModuleAttr : DialectAttribute<(attr
  ConcreteType<"ModuleType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def AnyOriginAttr : DialectAttribute<(attr
  ConcreteType<"OriginType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def ComptimeOriginAttr : DialectAttribute<(attr
  ConcreteType<"OriginType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def StaticOriginAttr : DialectAttribute<(attr
  ConcreteType<"OriginType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def WrappedOriginUnionAttr : DialectAttribute<(attr
  TypedAttrArray:$operands,
  ConcreteType<"OriginType">:$type
)>;

def OriginEqAttr : DialectAttribute<(attr
  TypedAttr:$lhs,
  TypedAttr:$rhs
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
  TypedAttr:$base,
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
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

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

def VarDeclKindAttr : DialectAttribute<(attr
  EnumAttr<"VarDeclKind">:$value
)>;

def ImplicitConversionKindAttr : DialectAttribute<(attr
  EnumAttr<"ImplicitConversionKind">:$value
)>;

def RefPackAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"RefPackType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def LITDialectAttributes : DialectAttributes<"lit"> {
  let elems = [
    DenseResourceElementsArrayAttr,
    SymbolAttr,
    EllipsisAttr,
    FnMetadataAttr,
    DocStringAttr,
    UnboundMLIROperationAttr,
    UnpackedAttr,
    ModuleAttr,
    AnyOriginAttr,
    OriginEqAttr,
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
    DeprecationInfoAttr,
    ComptimeOriginAttr,
    StaticOriginAttr,
    WrappedIndirectOriginAttr,
    ImplicitConversionKindAttr,
  ];
}

#endif // LITDIALECT_BYTECODE
