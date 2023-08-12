//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MDIALECT_BYTECODE
#define MDIALECT_BYTECODE

include "mlir/IR/BytecodeBase.td"

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

def StringAttrArray :
  WithParser<"succeeded($_reader.readAttributes($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeAttributes($_getter)",
  WithType<"SmallVector<StringAttr>">>>>;

def MultiLineStringAttr : DialectAttribute<(attr
  StringAttrArray:$value
)>;

def PointerAttr : DialectAttribute<(attr
  SignedVarInt:$addr,
  Type:$type
)>;

def MemoryBlobArray :
  WithParser<"succeeded(readMemoryBlobs($_reader, $_var))",
  WithBuilder<"$_args",
  WithPrinter<"writeMemoryBlobs($_writer, $_getter)",
  WithType<"SmallVector<MemoryBlob>">>>>;

def MemorySpaceAttr : DialectAttribute<(attr
  MemoryBlobArray:$value
)>;

def MemRefAttr : DialectAttribute<(attr
  MemorySpaceAttr:$memory,
  SignedVarInt:$index,
  SignedVarInt:$offset,
  Type:$type
)>;

def String :
  WithParser<"succeeded($_reader.readString($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeOwnedString($_getter)",
  WithType<"StringRef">>>>;

def LLVMTriple :
  WithParser<"succeeded(parseTriple($_reader, $_var))",
  WithBuilder<"$_args",
  WithPrinter<"printTriple($_name.getContext(), $_writer, $_getter)",
  WithType<"llvm::Triple">>>>;

def DataLayout :
  WithParser<"succeeded(parseDataLayout($_reader, $_var))",
  WithBuilder<"$_args",
  WithPrinter<"printDataLayout($_name.getContext(), $_writer, $_getter)",
  WithType<"DataLayout">>>>;

def TargetInfoAttr : DialectAttribute<(attr
  LLVMTriple:$triple,
  String:$cpu,
  String:$features,
  DataLayout:$data_layout,
  SignedVarInt:$simd_bit_width,
  String:$tune_cpu
)>;

def BuildInfoAttr : DialectAttribute<(attr
  String:$buildType,
  String:$kernelsBuildType,
  SignedVarInt:$lLCLMaxProfilingLevel
)>;

def MDialectAttributes : DialectAttributes<"M"> {
  let elems = [
    PrimitiveArrayAttr,
    ArrayElementsAttr,
    AlignedBytesAttr,
    MultiLineStringAttr,
    PointerAttr,
    MemorySpaceAttr,
    MemRefAttr,
    TargetInfoAttr,
    BuildInfoAttr
  ];
}

#endif // MDIALECT_BYTECODE
