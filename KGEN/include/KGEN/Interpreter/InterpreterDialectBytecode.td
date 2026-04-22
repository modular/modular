//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef INTERPRETERDIALECT_BYTECODE_TD
#define INTERPRETERDIALECT_BYTECODE_TD

include "Support/Compiler/Bytecode.td"

//===----------------------------------------------------------------------===//
// InterpreterDialect Attributes
//===----------------------------------------------------------------------===//

def PointerAttr : DialectAttribute<(attr
  SignedVarInt:$addr,
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def AlignedBlob :
  WithParser<"succeeded(readAlignedBlob($_reader, $_var))",
  WithPrinter<"writeAlignedBlob($_writer, $_getter)",
  WithType<"AlignedBlob">>>;

def MemoryHandleAttr : DialectAttribute<(attr
  AlignedBlob:$memory
)>;

def PointerRegions :
  WithParser<"succeeded(readPointerRegions($_reader, $_var))",
  WithPrinter<"writePointerRegions($_writer, $_getter)",
  WithType<"SmallVector<PointerRegion>">>>;

def SymbolRegions :
  WithParser<"succeeded(readSymbolRegions($_reader, $_var))",
  WithPrinter<"writeSymbolRegions($_writer, $_getter)",
  WithType<"SmallVector<int64_t>">>>;

def MemoryBlobAttr : DialectAttribute<(attr
  MemoryHandleAttr:$handle,
  EnumAttr<"MemoryKind">:$kind,
  PointerRegions:$pointerRegions,
  SymbolRegions:$symbolRegions,
  VarInt:$addressSpace
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def MemorySpaceAttr : DialectAttribute<(attr
  ConcreteAttrArray<"MemoryBlobAttr">:$value
)>;

def SymbolArrayAttr : DialectAttribute<(attr
  TypedAttrArray:$value
)>;

def MemoryModelAttr : DialectAttribute<(attr
  MemorySpaceAttr:$memory,
  SymbolArrayAttr:$symbols
)>;

def CoordinateAttr : DialectAttribute<(attr
  VarInt:$offset,
  VarInt:$refSlot,
  Type:$type
)>;

def MemRefAttr : DialectAttribute<(attr
  MemoryModelAttr:$model,
  SignedVarInt:$index,
  SignedVarInt:$offset,
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def StoreToMemAttr : DialectAttribute<(attr
  TypedAttr:$value,
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def UninitMemAttr : DialectAttribute<(attr
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def InterpreterDialectAttributes : DialectAttributes<"interp"> {
  let elems = [
    PointerAttr,
    MemoryHandleAttr,
    MemoryBlobAttr,
    MemorySpaceAttr,
    MemRefAttr,
    StoreToMemAttr,
    UninitMemAttr,
    MemoryModelAttr,
    SymbolArrayAttr,
    CoordinateAttr,
  ];
}

#endif // INTERPRETERDIALECT_BYTECODE_TD
