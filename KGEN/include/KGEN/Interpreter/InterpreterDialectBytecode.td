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

def PointerRegions :
  WithParser<"succeeded(readPointerRegions($_reader, $_var))",
  WithPrinter<"writePointerRegions($_writer, $_getter)",
  WithType<"SmallVector<PointerRegion>">>>;

def MemoryBlobAttr : DialectAttribute<(attr
  ResourceHandle<"MemoryHandle">:$handle,
  EnumAttr<"MemoryKind">:$kind,
  PointerRegions:$pointerRegions
)> {
  let cBuilder = "get<$_resultType>(context, *handle, kind, pointerRegions)";
}

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
    MemoryBlobAttr,
    MemorySpaceAttr,
    MemRefAttr,
    StoreToMemAttr
  ];
}

#endif // INTERPRETERDIALECT_BYTECODE_TD
