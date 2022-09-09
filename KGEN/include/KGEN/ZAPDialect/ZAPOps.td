//===- ZAPOps.td ----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ZAPOPS_TD
#define ZAPOPS_TD

include "KGEN/MetaDialect/MetaTypes.td"
include "KGEN/ZAPDialect/ZAPDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ZAPOp
//===----------------------------------------------------------------------===//

/// Base class for ZAP dialect operations.
class ZAPOp<string mnemonic, list<Trait> traits = []>
    : Op<ZAPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// BufferLoadOp
//===----------------------------------------------------------------------===//

def BufferLoadOp : ZAPOp<"buffer.load", [
     ScalarOfSameDType<"buffer", "result">]> {
  let summary = "buffer load operation";
  let description = [{
    The `buffer.load` operation takes a buffer and an index position and
    extracts the `!meta.scalar` value at that position. The position must be of
    type index.

    Example:

    ```mlir
    %1 = zap.buffer.load %0[%position]: !meta.buffer<size, dtype>
    ```
  }];

  let arguments = (ins Arg<ConcreteBuffer, "", [MemRead]>:$buffer,
                       Index:$position);
  let results = (outs ScalarType:$result);

  let assemblyFormat = "$buffer `[` $position `]` attr-dict `:` type($buffer)";
}

//===----------------------------------------------------------------------===//
// BufferStoreOp
//===----------------------------------------------------------------------===//

def BufferStoreOp : ZAPOp<"buffer.store", [
     ScalarOfSameDType<"buffer", "value">]> {
  let summary = "buffer store operation";
  let description = [{
    The `buffer.store.` operation takes a source value, a buffer, and an index
    position and inserts the source value into the buffer at that position. The
    position must be of type index.

    Example:

    ```mlir
    zap.buffer.store %val, %buf[%position]: !meta.buffer<size, dtype>
    ```
  }];

  let arguments = (ins Arg<ConcreteBuffer, "", [MemWrite]>:$buffer,
                       ScalarType:$value, Index:$position);

  let assemblyFormat = [{
    $value `,` $buffer `[` $position `]` attr-dict `:` type($buffer)
  }];
}


#endif // ZAPOPS_TD
