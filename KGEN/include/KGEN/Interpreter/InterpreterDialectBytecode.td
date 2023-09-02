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

def InterpreterDialectAttributes : DialectAttributes<"interp"> {
  let elems = [
    PointerAttr,
    MemorySpaceAttr,
    MemRefAttr
  ];
}

#endif // INTERPRETERDIALECT_BYTECODE_TD
