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

def MetaTypeType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  ConcreteAttribute<"ParamBindArrayAttr">:$paramValues
)>;

def LifetimeType : DialectType<(type)>;

def RefType : DialectType<(type
  EnumAttr<"bool">:$isMutable,
  TypedAttr:$elementType,
  TypedAttr:$lifetime
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
    MetaTypeType,
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

def FnMetadataAttr : DialectAttribute<(attr
  StringAttrArray:$argNames,
  EnumAttrArray<"PassingKind">:$argPassingKinds,
  StringAttrArray:$paramNames,
  EnumAttrArray<"PassingKind">:$paramPassingKinds,
  TypedAttrArray:$defaultArguments,
  TypedAttrArray:$defaultParameters
)>;

def DocStringAttr : DialectAttribute<(attr
  ConcreteAttribute<"StringAttr">:$string,
  OptionalAttribute<"FileLineColLoc">:$location
)>;

def UnboundMLIROperationAttr : DialectAttribute<(attr
  ConcreteAttribute<"StringAttr">:$name,
  ConcreteAttribute<"DictionaryAttr">:$attrs
)>;

def ModuleAttr : DialectAttribute<(attr
  ConcreteType<"MetaTypeType">:$type
)>;

def LifetimeAttr : DialectAttribute<(attr)>;

def UnresolvedAliasValueAttr : DialectAttribute<(attr
  Type:$type
)>;

def LITStructAttr : DialectAttribute<(attr
  WithParser<"succeeded(readStructValues($_reader, $_var))",
  WithPrinter<"writeStructValues($_writer, $_getter)",
  WithType<"SmallVector<std::tuple<StringAttr, TypedAttr>>">>>:$values,
  ConcreteType<"DeclRefType">:$type
)>;

def PassingKindArrayAttr : DialectAttribute<(attr
  EnumAttrArray<"PassingKind">:$value
)>;

def LITDialectAttributes : DialectAttributes<"lit"> {
  let elems = [
    FnMetadataAttr,
    DocStringAttr,
    UnboundMLIROperationAttr,
    ModuleAttr,
    LifetimeAttr,
    UnresolvedAliasValueAttr,
    LITStructAttr,
    PassingKindArrayAttr,
    /*StructExtractAttr*/
  ];
}

#endif // LITDIALECT_BYTECODE
