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

def ParamRefType : DialectType<(type
  TypedAttr:$param
)>;

def TypeType : DialectType<(type)>;

def DTypeType : DialectType<(type)>;

def StringType : DialectType<(type)>;

def IntLiteralType : DialectType<(type)>;

def SignatureType : DialectType<(type
  TypeArray:$inputParamTypes,
  TypeArray:$resultParamTypes,
  ConcreteType<"FunctionType">:$values,
  EnumAttrArray<"ValueInputConvention">:$inputConventions,
  Custom<"FnEffects">:$fnEffects,
  ConcreteOptionalAttribute<"FnMetadataAttrInterface">:$metadata
)>;

def PointerType : DialectType<(type
  Type:$elementType,
  TypedAttr:$addressSpace
)>;

def DeclRefType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  Type:$metaType
)>;

def VariadicType : DialectType<(type
  Type:$elementType,
  EnumAttr<"ValueInputConvention">:$convention
)>;

def NoneType : DialectType<(type)>;

def TargetType : DialectType<(type)>;

def BuildInfoType : DialectType<(type)>;

def StructType : DialectType<(type
  TypeArray:$elementTypes,
  Bool:$isMemoryOnly
)>;

def PackType : DialectType<(type
  TypedAttr:$variadic
)>;

def VariantType : DialectType<(type
  TypedAttrArray:$types
)>;

def KGENDialectTypes : DialectTypes<"kgen"> {
  let elems = [
    ParamRefType,
    TypeType,
    DTypeType,
    StringType,
    IntLiteralType,
    SignatureType,
    PointerType,
    DeclRefType,
    VariadicType,
    NoneType,
    TargetType,
    BuildInfoType,
    StructType,
    PackType,
    VariantType
  ];
}

//===----------------------------------------------------------------------===//
// KGENDialect Attributes
//===----------------------------------------------------------------------===//

def ParameterExprArrayAttr : DialectAttribute<(attr
  TypedAttrArray:$value
)>;

def ParamDeclAttr : DialectAttribute<(attr
  StringAttr:$name,
  Type:$type
)>;

def ParamDeclArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"ParamDeclAttr">:$value
)>;

def ConstraintAttr : DialectAttribute<(attr
  TypedAttr:$expr,
  StringAttr:$message,
  ConcreteAttribute<"mlir::LocationAttr">:$loc
)>;

def ConstraintArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"ConstraintAttr">:$value
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

def ConcreteTypeConstantAttr : DialectAttribute<(attr
  Type:$value,
  Type:$type,
  ConcreteAttribute<"VTableAttr">:$vTable
)>;

def ParameterizedTypeConstantAttr : DialectAttribute<(attr
  Type:$value,
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

def ParamOperatorAttr : DialectAttribute<(attr
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

def PackageArchiveAttr : DialectAttribute<(attr
  ConcreteAttribute<"TargetInfoAttr">:$target,
  ConcreteAttribute<"DenseResourceElementsAttr">:$elaboratedModule,
  ConcreteAttribute<"DenseResourceElementsAttr">:$archive,
  ConcreteOptionalAttribute<"LinkDependencyArrayAttr">:$dependencies
)>;

def PackageArchiveArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"PackageArchiveAttr">:$value
)>;

def StructAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"StructType">:$type
)>;

def StructExtractAttr : DialectAttribute<(attr
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

def LinkDependencyAttr : DialectAttribute<(attr
  StringAttr:$name,
  ConcreteAttribute<"DenseResourceElementsAttr">:$bytes
)>;

def LinkDependencyArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"LinkDependencyAttr">:$value
)>;

def KGENDialectAttributes : DialectAttributes<"kgen"> {
  let elems = [
    ParameterExprArrayAttr,
    ParamDeclAttr,
    ParamDeclArrayAttr,
    ConstraintAttr,
    ConstraintArrayAttr,
    VariadicAttr,
    UnknownAttr,
    UnboundAttr,
    NoneAttr,
    ParamDeclRefAttr,
    ParamIndexRefAttr,
    VTableAttr,
    VTableEntryAttr,
    ConcreteTypeConstantAttr,
    ParameterizedTypeConstantAttr,
    DTypeConstantAttr,
    IntLiteralAttr,
    SymbolConstantAttr,
    TargetParamAttr,
    EnvAttr,
    ParamOperatorAttr,
    MLIROpAttr,
    DecoratorsAttr,
    ExportKindAttr,
    PackageArchiveAttr,
    PackageArchiveArrayAttr,
    StructAttr,
    StructExtractAttr,
    PackAttr,
    VariantAttr,
    LinkDependencyAttr,
    LinkDependencyArrayAttr
  ];
}

#endif // KGENDIALECT_BYTECODE
