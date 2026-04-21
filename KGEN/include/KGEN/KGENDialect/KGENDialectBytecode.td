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

def NonStructTypeType : DialectType<(type)>;

def TypeType : DialectType<(type)>;

def DTypeType : DialectType<(type)>;

def StringType : DialectType<(type)>;

def GeneratorType : DialectType<(type
  TypeArray:$inputParamTypes,
  Type:$body,
  ConcreteOptionalAttribute<"GeneratorMetadataAttrInterface">:$metadata
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def FuncType : DialectType<(type
  ConcreteType<"FunctionType">:$values,
  EnumAttrArray<"ArgConvention">:$argConventions,
  Custom<"FnEffects">:$fnEffects,
  ConcreteOptionalAttribute<"FnMetadataAttrInterface">:$metadata
)>;

def FuncLiteralType : DialectType<(type
  TypedAttr:$funcLiteral
)>;

def PointerType : DialectType<(type
  Type:$elementType,
  TypedAttr:$addressSpace,
  Bool:$isNonNull
)>;

def ParamListType : DialectType<(type
  Type:$elementType
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def NoneType : DialectType<(type)>;

def TargetType : DialectType<(type)>;

def BuildInfoType : DialectType<(type)>;

def StructType : DialectType<(type
  TypedAttr:$elementTypesVariadic,
  TypedAttr:$isMemoryOnly,
  TypedAttr:$minAlignment
)>;

def StructInstanceType : DialectType<(type
  StringAttr:$name,
  ConcreteAttrArray<"StringAttr">:$paramNames,
  TypedAttrArray:$paramValues,
  ConcreteAttrArray<"StructDefFieldAttr">:$fields,
  TypedAttr:$isMemoryOnly
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def PackType : DialectType<(type
  TypedAttr:$variadic
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def WrappedVariantType : DialectType<(type
  TypedAttr:$variadic
)>;

def ClosureType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$parentSymbol,
  StringAttr:$name,
  EnumAttr<"ClosureMemoryKind">:$closureMemoryKind
)>;

def ParamClosureType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$parentSymbol,
  StringAttr:$name
)>;

def KGENDialectTypes : DialectTypes<"kgen"> {
  let elems = [
    WrappedParamRefType,
    WrappedTypeValueType,
    NonStructTypeType,
    TypeType,
    DTypeType,
    StringType,
    GeneratorType,
    FuncType,
    FuncLiteralType,
    PointerType,
    ParamListType,
    NoneType,
    TargetType,
    BuildInfoType,
    StructType,
    StructInstanceType,
    PackType,
    WrappedVariantType,
    ClosureType,
    ParamClosureType,
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

def ParamListAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"ParamListType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def ParamListReduceAttr : DialectAttribute<(attr
    Type:$type,
    TypedAttr:$base,
    TypedAttr:$paramList,
    TypedAttr:$generator
)>;

def ParamListTabulateAttr : DialectAttribute<(attr
  ConcreteType<"ParamListType">:$type,
  TypedAttr:$count,
  TypedAttr:$generator
)>;

def ParamListSizeAttr : DialectAttribute<(attr
  ConcreteType<"IndexType">:$type,
  TypedAttr:$paramList
)>;

def ParamListGetAttr : DialectAttribute<(attr
  Type:$type,
  TypedAttr:$paramList,
  TypedAttr:$index
)>;

def ParamListConcatAttr : DialectAttribute<(attr
  ConcreteType<"ParamListType">:$type,
  TypedAttr:$paramLists
)>;

def UnknownAttr : DialectAttribute<(attr
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def UnboundAttr : DialectAttribute<(attr
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def NoneAttr : DialectAttribute<(attr)>;

def ParamDeclRefAttr : DialectAttribute<(attr
  StringAttr:$name,
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def ParamIndexRefAttr : DialectAttribute<(attr
  VarInt:$depth,
  VarInt:$index,
  Type:$type
)>;

def WrappedTypeParamAttr : DialectAttribute<(attr
  Type:$typeValue,
  Type:$mlirType,
  Type:$type
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
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def TypeInstanceRefAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DTypeConstantAttr : DialectAttribute<(attr
  Custom<"KGENDType">:$dType
)>;

def SymbolConstantAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  ConcreteType<"GeneratorType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def FuncSymbolAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  ConcreteType<"FuncType">:$type
)>;

def GeneratorAttr : DialectAttribute<(attr
  TypeArray:$inputParamTypes,
  ConcreteAttribute<"TypedAttr">:$body,
  ConcreteOptionalAttribute<"GeneratorMetadataAttrInterface">:$metadata
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def TargetParamAttr : DialectAttribute<(attr
  ConcreteAttribute<"TargetInfoAttr">:$target
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

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
  TypedAttr:$typeValue
)>;

def StructAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"StructType">:$type
)>;

def WrappedStructExtractAttr : DialectAttribute<(attr
  TypedAttr:$structValue,
  TypedAttr:$fieldNo,
  Type:$type
)>;

def PackAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"PackType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def UpcastAttr : DialectAttribute<(attr
  Type:$type,
  TypedAttr:$inputTypeValue
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
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def LinkDependencyArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"FlatSymbolRefAttr">:$value
)>;

def PreservedAttr : DialectAttribute<(attr
  Attribute:$value
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def TailKindAttr : DialectAttribute<(attr
  EnumAttr<"TailKind">:$value
)>;

def ClosureMethodAttr : DialectAttribute<(attr
  EnumAttr<"ClosureMethod">:$value
)>;

def ConstraintAttr : DialectAttribute<(attr
  TypedAttr:$prop,
  ConcreteAttribute<"LocationAttr">:$loc
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def LLVMBitcodeLibAttr : DialectAttribute<(attr
  ConcreteAttribute<"BoolAttr">:$used,
  Attribute:$library
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def LLVMBitcodeLibArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"LLVMBitcodeLibAttr">:$value
)>;

def SugarAttr : DialectAttribute<(attr
  EnumAttr<"SugarKind">:$kind,
  ConcreteOptionalAttribute<"StringAttr">:$memberName,
  TypedAttr:$sugared,
  TypedAttr:$expanded
)>;

def ClosureSymbolAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$parentSymbol,
  StringAttr:$nestedFuncName,
  ConcreteAttribute<"ClosureMethodAttr">:$method,
  TypedAttrArray:$paramValues,
  ConcreteType<"GeneratorType">:$type
)>;

def ClosureAttr : DialectAttribute<(attr
  ConcreteType<"ParamClosureType">:$type
)>;

def MemSymbolTripleAttr : DialectAttribute<(attr
  OptionalAttribute<"TypedAttr">:$copy,
  OptionalAttribute<"TypedAttr">:$move,
  OptionalAttribute<"TypedAttr">:$del,
  ConcreteOptionalAttribute<"UnitAttr">:$isMove
)>;

def LinkageNameAttr : DialectAttribute<(attr
  TypedAttr:$name,
  ConcreteAttribute<"BoolAttr">:$mangle
)>;

def KGENDialectAttributes : DialectAttributes<"kgen"> {
  let elems = [
    ParameterExprArrayAttr,
    ParamDeclAttr,
    ParamDeclArrayAttr,
    ParamListAttr,
    ParamListReduceAttr,
    ParamListTabulateAttr,
    ParamListSizeAttr,
    ParamListGetAttr,
    ParamListConcatAttr,
    UnknownAttr,
    UnboundAttr,
    NoneAttr,
    ParamDeclRefAttr,
    ParamIndexRefAttr,
    WrappedTypeParamAttr,
    BindParamsAttr,
    TypeGeneratorRefAttr,
    TypeInstanceRefAttr,
    DTypeConstantAttr,
    SymbolConstantAttr,
    FuncSymbolAttr,
    GeneratorAttr,
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
    TailKindAttr,
    LLVMBitcodeLibAttr,
    LLVMBitcodeLibArrayAttr,
    SugarAttr,
    ClosureSymbolAttr,
    ClosureAttr,
    ClosureMethodAttr,
    MemSymbolTripleAttr,
    LinkageNameAttr
  ];
}

#endif // KGENDIALECT_BYTECODE
