//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGENDIALECT_BYTECODE
#define KGENDIALECT_BYTECODE

include "Support/Compiler/Bytecode.td"

//===----------------------------------------------------------------------===//
// KGENDialect Helper Attributes
//===----------------------------------------------------------------------===//

def ParamDeclAttr : DialectAttribute<(attr
  StringAttr:$name,
  Type:$type
)>;

def ParamDeclArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"ParamDeclAttr">:$value
)>;

def StructDefFieldAttr : DialectAttribute<(attr
  StringAttr:$name,
  Type:$type
)>;

def StructDefAttr : DialectType<(type
  StringAttr:$name,
  ConcreteAttrArray<"StringAttr">:$inputParamNames,
  TypeArray:$inputParamTypes,
  ConcreteAttrArray<"StructDefFieldAttr">:$fields,
  Bool:$isMemoryOnly
)>;

//===----------------------------------------------------------------------===//
// KGENDialect Types
//===----------------------------------------------------------------------===//

def WrappedParamRefType : DialectType<(type
  TypedAttr:$param
)>;

def WrappedTypeValueType : DialectType<(type
  TypedAttr:$typeValue
)>;

def TypeType : DialectType<(type)>;

def DTypeType : DialectType<(type)>;

def StringType : DialectType<(type)>;

def IntLiteralType : DialectType<(type)>;

def SignatureType : DialectType<(type
  TypeArray:$inputParamTypes,
  TypeArray:$resultParamTypes,
  ConcreteType<"FunctionType">:$values,
  EnumAttrArray<"ArgConvention">:$argConventions,
  Custom<"FnEffects">:$fnEffects,
  ConcreteOptionalAttribute<"FnMetadataAttrInterface">:$metadata
)>;

def PointerType : DialectType<(type
  Type:$elementType,
  TypedAttr:$addressSpace,
  TypedAttr:$exclusive
)>;

def VariadicType : DialectType<(type
  Type:$elementType,
  EnumAttr<"ArgConvention">:$convention
)>;

def NoneType : DialectType<(type)>;

def TargetType : DialectType<(type)>;

def BuildInfoType : DialectType<(type)>;

def StructType : DialectType<(type
  TypeArray:$elementTypes,
  Bool:$isMemoryOnly
)>;

def AppliedStructType : DialectType<(type
  StructDefAttr:$sourceStruct,
  TypedAttrArray:$paramValues
)>;

def PackType : DialectType<(type
  TypedAttr:$variadic
)>;

def WrappedVariantType : DialectType<(type
  TypedAttr:$variadic
)>;

def FloatLiteralType : DialectType<(type)>;

def KGENDialectTypes : DialectTypes<"kgen"> {
  let elems = [
    WrappedParamRefType,
    WrappedTypeValueType,
    TypeType,
    DTypeType,
    StringType,
    IntLiteralType,
    SignatureType,
    PointerType,
    VariadicType,
    NoneType,
    TargetType,
    BuildInfoType,
    StructType,
    PackType,
    WrappedVariantType,
    FloatLiteralType
  ];
}

//===----------------------------------------------------------------------===//
// KGENDialect Attributes
//===----------------------------------------------------------------------===//

def ParameterExprArrayAttr : DialectAttribute<(attr
  TypedAttrArray:$value
)>;

def VariadicAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"VariadicType">:$type
)>;

def UnknownAttr : DialectAttribute<(attr
  Type:$type
)>;

def UnboundAttr : DialectAttribute<(attr
  Type:$type
)>;

def NoneAttr : DialectAttribute<(attr)>;

def ParamDeclRefAttr : DialectAttribute<(attr
  StringAttr:$name,
  Type:$type
)>;

def ParamIndexRefAttr : DialectAttribute<(attr
  VarInt:$depth,
  Bool:$isResult,
  VarInt:$index,
  Type:$type
)>;

def VTableAttr : DialectAttribute<(attr
  ConcreteAttrArray<"VTableEntryAttr">:$entries
)>;

def VTableEntryAttr : DialectAttribute<(attr
  StringAttr:$name,
  ConcreteAttribute<"TypedAttr">:$method
)>;

def WrappedTypeConstantAttr : DialectAttribute<(attr
  Type:$typeValue,
  Type:$mlirType,
  Type:$type,
  ConcreteAttribute<"VTableAttr">:$vTable
)>;

def DTypeConstantAttr : DialectAttribute<(attr
  Custom<"KGENDType">:$dType
)>;

def IntLiteralAttr : DialectAttribute<(attr
  Custom<"IPInt">:$value
)>;

def SymbolConstantAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  ConcreteType<"SignatureType">:$type
)>;

def TargetParamAttr : DialectAttribute<(attr
  ConcreteAttribute<"TargetInfoAttr">:$target
)>;

def EnvAttr : DialectAttribute<(attr
  ConcreteAttribute<"DictionaryAttr">:$values
)>;

def WrappedParamOperatorAttr : DialectAttribute<(attr
  EnumAttr<"POC">:$opcode,
  TypedAttrArray:$operands,
  Type:$type
)>;

def MLIROpAttr : DialectAttribute<(attr
  StringAttr:$name,
  ConcreteAttribute<"DictionaryAttr">:$attrs,
  ConcreteType<"SignatureType">:$type
)>;

def DecoratorsAttr : DialectAttribute<(attr
  TypedAttrArray:$value
)>;

def ExportKindAttr : DialectAttribute<(attr
  EnumAttr<"ExportKind">:$value
)>;

def StructDefParamRefAttr : DialectAttribute<(attr
  StringAttr:$name,
  Type:$type
)>;

def StructAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"StructType">:$type
)>;

def WrappedStructExtractAttr : DialectAttribute<(attr
  TypedAttr:$structValue,
  VarInt:$fieldNo,
  Type:$type
)>;

def PackAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"PackType">:$type
)>;

def VariantAttr : DialectAttribute<(attr
  TypedAttr:$value,
  VarInt:$index,
  ConcreteType<"VariantType">:$type
)>;

def LinkDependencyArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"FlatSymbolRefAttr">:$value
)>;

def FloatLiteralAttr : DialectAttribute<(attr
  ConcreteAttribute<"FloatLiteralSpecialValuesAttr">:$special,
  Custom<"OptionalIPRational">:$rational
)>;

def PreservedAttr : DialectAttribute<(attr
  Attribute:$value
)>;

def KGENDialectAttributes : DialectAttributes<"kgen"> {
  let elems = [
    ParamDeclAttr,
    ParamDeclArrayAttr,
    StructDefFieldAttr,
    ParameterExprArrayAttr,
    VariadicAttr,
    UnknownAttr,
    UnboundAttr,
    NoneAttr,
    ParamDeclRefAttr,
    ParamIndexRefAttr,
    VTableAttr,
    VTableEntryAttr,
    WrappedTypeConstantAttr,
    DTypeConstantAttr,
    IntLiteralAttr,
    SymbolConstantAttr,
    TargetParamAttr,
    EnvAttr,
    WrappedParamOperatorAttr,
    MLIROpAttr,
    DecoratorsAttr,
    ExportKindAttr,
    StructAttr,
    WrappedStructExtractAttr,
    PackAttr,
    VariantAttr,
    LinkDependencyArrayAttr,
    FloatLiteralAttr,
    PreservedAttr
  ];
}

#endif // KGENDIALECT_BYTECODE
