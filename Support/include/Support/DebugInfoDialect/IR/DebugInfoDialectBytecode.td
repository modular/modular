//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef DEBUGINFODIALECT_BYTECODE
#define DEBUGINFODIALECT_BYTECODE

include "Support/Compiler/Bytecode.td"

//===----------------------------------------------------------------------===//
// DebugInfoDialect Types
//===----------------------------------------------------------------------===//

def DIArrayType : DialectType<(type
  ConcreteType<"DIType">:$elementType,
  VarInt:$elementCount
)>;

def DIBasicType : DialectType<(type
  StringAttr:$name,
  VarInt:$sizeInBits,
  VarInt:$alignInBits,
  VarInt:$encoding
)>;

def DIMemberType : DialectType<(type
  StringAttr:$name,
  ConcreteType<"DIType">:$type
)>;

def DIPointerType : DialectType<(type
  ConcreteType<"DIType">:$elementType,
  VarInt:$sizeInBits,
  VarInt:$alignInBits,
  Custom<"OptionalUnsigned">:$addressSpace
)>;

def DIStructType : DialectType<(type
  StringAttr:$name,
  ConcreteTypeArray<"DIMemberType">:$members
)>;

def DISubroutineType : DialectType<(type
  VarInt:$callingConvention,
  ConcreteTypeArray<"DIType">:$argumentTypes,
  ConcreteTypeArray<"DIType">:$resultTypes
)>;

def DIUnresolvedMLIRType : DialectType<(type
  Type:$type
)>;

def DIUnspecifiedType : DialectType<(type
  StringAttr:$name
)>;

def DIVectorType : DialectType<(type
  ConcreteType<"DIType">:$elementType,
  VarInt:$elementCount
)>;

def DebugInfoDialectTypes : DialectTypes<"debuginfo"> {
  let elems = [
    DIArrayType,
    DIBasicType,
    DIMemberType,
    DIPointerType,
    DIStructType,
    DISubroutineType,
    DIUnresolvedMLIRType,
    DIUnspecifiedType,
    DIVectorType
  ];
}

//===----------------------------------------------------------------------===//
// DebugInfoDialect Attributes
//===----------------------------------------------------------------------===//

def SourceNameAttr : DialectAttribute<(attr
  StringAttr:$name,
  ConcreteAttrArray<"SourceNameAttr">:$paramTypes,
  ConcreteAttrArray<"SourceNameAttr">:$argTypes,
  StringAttrArray:$paramValues,
  ConcreteOptionalAttribute<"SourceNameAttr">:$parent,
  EnumAttr<"SourceNameKind">:$kind,
  ConcreteAttrArray<"SourceNameAttr">:$decorators
)>;

def DICompileUnitAttr : DialectAttribute<(attr
  VarInt:$sourceLanguage,
  ConcreteAttribute<"DIFileAttr">:$file,
  StringAttr:$producer,
  Bool:$isOptimized,
  EnumAttr<"EmissionKind">:$emissionKind,
  EnumAttr<"NameTableKind">:$nameTableKind
)>;

def DIFileAttr : DialectAttribute<(attr
  StringAttr:$name,
  StringAttr:$directory
)>;

def DILexicalBlockAttr : DialectAttribute<(attr
  ConcreteAttribute<"DILocalScopeAttr">:$scope,
  ConcreteAttribute<"DIFileAttr">:$file,
  VarInt:$line,
  VarInt:$column
)>;

def DILocalVariableAttr : DialectAttribute<(attr
  ConcreteAttribute<"DILocalScopeAttr">:$scope,
  StringAttr:$name,
  ConcreteOptionalAttribute<"DIFileAttr">:$file,
  VarInt:$line,
  VarInt:$arg,
  VarInt:$alignInBits,
  ConcreteType<"DIType">:$type,
  EnumAttr<"DIFlags">:$flags
)>;

def DISubprogramAttr : DialectAttribute<(attr
  ConcreteOptionalAttribute<"DICompileUnitAttr">:$compileUnit,
  ConcreteOptionalAttribute<"DIScopeAttr">:$scope,
  ConcreteAttribute<"SourceNameAttr">:$name,
  ConcreteOptionalAttribute<"StringAttr">:$linkageName,
  ConcreteOptionalAttribute<"DIFileAttr">:$file,
  VarInt:$line,
  VarInt:$scopeLine,
  EnumAttr<"SubprogramFlags">:$subprogramFlags,
  ConcreteType<"DISubroutineType">:$type
)>;

def DebugInfoDialectAttributes : DialectAttributes<"debuginfo"> {
  let elems = [
    SourceNameAttr,
    DICompileUnitAttr,
    DIFileAttr,
    DILexicalBlockAttr,
    DILocalVariableAttr,
    DISubprogramAttr
  ];
}

#endif // DEBUGINFODIALECT_BYTECODE
