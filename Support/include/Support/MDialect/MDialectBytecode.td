//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MDIALECT_BYTECODE
#define MDIALECT_BYTECODE

include "Support/Compiler/Bytecode.td"

//===----------------------------------------------------------------------===//
// MDialect Types
//===----------------------------------------------------------------------===//

def ArrayType : DialectType<(type
  SignedVarInt:$size,
  Type:$elementType
)>;

def AlignedBytesType : DialectType<(type
  VarInt:$size,
  VarInt:$align
)>;

def MDialectTypes : DialectTypes<"M"> {
  let elems = [ArrayType, AlignedBytesType];
}

//===----------------------------------------------------------------------===//
// MDialect Attributes
//===----------------------------------------------------------------------===//

def StringArrayAttr : DialectAttribute<(attr
  StringAttrArray:$value
)>;

def TypeArrayAttr : DialectAttribute<(attr
  TypeArray:$value
)>;

def SignlessBytes :
  WithParser<"succeeded($_reader.readBlob($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeOwnedBlob({(const char *)$_name.getData().data(),"
                                        "$_name.getData().size()})",
  WithType<"ArrayRef<char>">>>>;

def ToUnsignedBytes :
  LocalVar<"ArrayRef<uint8_t>",
           "{(const uint8_t *)_charData.data(), _charData.size()}">;

def PrimitiveArrayAttr : DialectAttribute<(attr
  SignlessBytes:$_charData,
  ToUnsignedBytes:$data,
  Type:$elementType
)>;

def ArrayElementsAttr : DialectAttribute<(attr
  PrimitiveArrayAttr:$data,
  WithType<"ShapedType", Type>:$type
)>;

def AlignedBytesAttr : DialectAttribute<(attr
  SignlessBytes:$_charData,
  ToUnsignedBytes:$data,
  VarInt:$align
)>;

def MultiLineStringAttr : DialectAttribute<(attr
  StringAttrArray:$value
)>;

def LLVMTriple :
  WithParser<"succeeded(parseTriple($_reader, $_var))",
  WithBuilder<"$_args",
  WithPrinter<"printTriple($_name.getContext(), $_writer, $_getter)",
  WithType<"llvm::Triple">>>>;

def DataLayout : Custom<"DataLayout">;

def LLVMRelocModel :
  WithParser<"succeeded(parseRelocModel($_reader, $_var))",
  WithBuilder<"$_args",
  WithPrinter<"printRelocModel($_name.getContext(), $_writer, $_getter)",
  WithType<"llvm::Reloc::Model">>>>;

def OptionalInt64 :
  WithParser<"succeeded(readOptional($_reader, $_var, &readIntegral<int64_t>))",
  WithPrinter<"writeOptional($_writer, $_getter, &writeIntegral<int64_t>)",
  WithType<"std::optional<int64_t>">>>;

def TargetInfoAttr : DialectAttribute<(attr
  LLVMTriple:$triple,
  String:$arch,
  String:$features,
  Custom<"DataLayout">:$data_layout,
  LLVMRelocModel:$relocation_model,
  SignedVarInt:$simd_bit_width,
  OptionalInt64:$index_bit_width,
  String:$tune_cpu,
  String:$accelerator_arch
)>;

def MDialectAttributes : DialectAttributes<"M"> {
  let elems = [
    StringArrayAttr,
    TypeArrayAttr,
    PrimitiveArrayAttr,
    ArrayElementsAttr,
    AlignedBytesAttr,
    MultiLineStringAttr,
    TargetInfoAttr
  ];
}

#endif // MDIALECT_BYTECODE
