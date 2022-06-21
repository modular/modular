//===- MetaOps.td ---------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the Meta dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// MetaBufferSizeOp
//===----------------------------------------------------------------------===//

def MetaBufferSizeOp : MetaOp<"buffer.size", [NoSideEffect]> {

  let summary = [{
    Return the size of a meta.buffer as an SSA value (of type `index`) that may
    be used by other ops.

    Example:

    ```mlir
    %0 = meta.buffer.size %arg0 : meta.buffer<3xf32>
    ```
  }];

  let arguments = (ins BufferType:$value);
  let results = (outs Index:$result);
  let assemblyFormat = "$value attr-dict `:` qualified(type($value))";
  let hasFolder = true;
}
