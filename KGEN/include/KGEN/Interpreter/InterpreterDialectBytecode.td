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
)>;

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

def MemoryBlobAttr : DialectAttribute<(attr
  MemoryHandleAttr:$handle,
  EnumAttr<"MemoryKind">:$kind,
  PointerRegions:$pointerRegions
)>;

def MemorySpaceAttr : DialectAttribute<(attr
  ConcreteAttrArray<"MemoryBlobAttr">:$value
)>;

def MemRefAttr : DialectAttribute<(attr
  MemorySpaceAttr:$memory,
  SignedVarInt:$index,
  SignedVarInt:$offset,
  Type:$type
)>;

def StoreToMemAttr : DialectAttribute<(attr
  TypedAttr:$value,
  Type:$type
)>;

def InterpreterDialectAttributes : DialectAttributes<"interp"> {
  let elems = [
    PointerAttr,
    MemoryHandleAttr,
    MemoryBlobAttr,
    MemorySpaceAttr,
    MemRefAttr,
    StoreToMemAttr
  ];
}

#endif // INTERPRETERDIALECT_BYTECODE_TD
