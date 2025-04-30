//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGENDIALECT_BYTECODE
#define KGENDIALECT_BYTECODE

include "Support/Compiler/Bytecode.td"

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

def GeneratorType : DialectType<(type
  TypeArray:$inputParamTypes,
  Type:$body,
  ConcreteOptionalAttribute<"GeneratorMetadataAttrInterface">:$metadata
)>;

def FuncType : DialectType<(type
  ConcreteType<"FunctionType">:$values,
  EnumAttrArray<"ArgConvention">:$argConventions,
  Custom<"FnEffects">:$fnEffects,
  ConcreteOptionalAttribute<"FnMetadataAttrInterface">:$metadata
)>;

def PointerType : DialectType<(type
  Type:$elementType,
  TypedAttr:$addressSpace
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

def StructInstanceType : DialectType<(type
  StringAttr:$name,
  ConcreteAttrArray<"StringAttr">:$paramNames,
  TypedAttrArray:$paramValues,
  ConcreteAttrArray<"StructDefFieldAttr">:$fields,
  Bool:$isMemoryOnly
)>;

def PackType : DialectType<(type
  TypedAttr:$variadic
)>;

def WrappedVariantType : DialectType<(type
  TypedAttr:$variadic
)>;

def KGENDialectTypes : DialectTypes<"kgen"> {
  let elems = [
    WrappedParamRefType,
    WrappedTypeValueType,
    TypeType,
    DTypeType,
    StringType,
    GeneratorType,
    FuncType,
    PointerType,
    VariadicType,
    NoneType,
    TargetType,
    BuildInfoType,
    StructType,
    StructInstanceType,
    PackType,
    WrappedVariantType,
  ];
}

//===----------------------------------------------------------------------===//
// KGENDialect Attributes
//===----------------------------------------------------------------------===//

def ParamDeclAttr : DialectAttribute<(attr
  StringAttr:$name,
  Type:$type
)>;

def ParamDeclArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"ParamDeclAttr">:$value
)>;

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

def BindParamsAttr : DialectAttribute<(attr
  ConcreteAttribute<"TypedAttr">:$generator,
  TypedAttrArray:$paramValues,
  Type:$type
)>;

def TypeGeneratorRefAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  Type:$type
)>;

def TypeInstanceRefAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  Type:$type
)>;

def DTypeConstantAttr : DialectAttribute<(attr
  Custom<"KGENDType">:$dType
)>;

def SymbolConstantAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  ConcreteType<"GeneratorType">:$type
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
  ConcreteType<"FuncTypeGeneratorType">:$type
)>;

def DecoratorsAttr : DialectAttribute<(attr
  TypedAttrArray:$value
)>;

def ExportKindAttr : DialectAttribute<(attr
  EnumAttr<"ExportKind">:$value
)>;

def StructDefFieldAttr : DialectAttribute<(attr
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

def UpcastAttr : DialectAttribute<(attr
  Type:$type,
  TypedAttr:$inputTypeValue,
  ConcreteAttribute<"VTableAttr">:$vTable
)>;

def GetWitnessAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  StringAttr:$traitName,
  StringAttr:$witnessName,
  Type:$type
)>;

def VariantAttr : DialectAttribute<(attr
  TypedAttr:$value,
  VarInt:$index,
  ConcreteType<"VariantType">:$type
)>;

def LinkDependencyArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"FlatSymbolRefAttr">:$value
)>;

def PreservedAttr : DialectAttribute<(attr
  Attribute:$value
)>;

def TailKindAttr : DialectAttribute<(attr
  EnumAttr<"TailKind">:$value
)>;

def KGENDialectAttributes : DialectAttributes<"kgen"> {
  let elems = [
    ParameterExprArrayAttr,
    ParamDeclAttr,
    ParamDeclArrayAttr,
    VariadicAttr,
    UnknownAttr,
    UnboundAttr,
    NoneAttr,
    ParamDeclRefAttr,
    ParamIndexRefAttr,
    VTableAttr,
    VTableEntryAttr,
    WrappedTypeConstantAttr,
    BindParamsAttr,
    TypeGeneratorRefAttr,
    TypeInstanceRefAttr,
    DTypeConstantAttr,
    SymbolConstantAttr,
    TargetParamAttr,
    StructDefFieldAttr,
    StructAttr,
    WrappedStructExtractAttr,
    PackAttr,
    UpcastAttr,
    GetWitnessAttr,
    VariantAttr,
    EnvAttr,
    WrappedParamOperatorAttr,
    MLIROpAttr,
    DecoratorsAttr,
    PreservedAttr,
    ExportKindAttr,
    LinkDependencyArrayAttr,
    TailKindAttr
  ];
}

#endif // KGENDIALECT_BYTECODE
