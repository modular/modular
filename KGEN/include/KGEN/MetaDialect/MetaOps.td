//===- MetaOps.td ---------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the Meta dialect.
//
//===----------------------------------------------------------------------===//

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

def MetaBufferCastOp : MetaOp<"buffer.cast", [NoSideEffect]> {
  let summary = [{
    Cast the input buffer to a buffer of the specified type.

    Example:

    ```mlir
    %0 = meta.buffer.cast %unknown: !buffer<?x?> to !buffer<?xf32>
    %1 = meta.buffer.cast %other: !buffer<1023xf32> to !buffer<?xf32>
  }];

  let arguments = (ins BufferType:$buffer);
  let results = (outs BufferType:$result);
  let assemblyFormat = "$buffer `:` qualified(type($buffer)) `to` qualified(type($result)) attr-dict";
  let hasVerifier = true;
}
