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

//===----------------------------------------------------------------------===//
// ZAPOp
//===----------------------------------------------------------------------===//

/// Base class for ZAP dialect operations.
class ZAPOp<string mnemonic, list<Trait> traits = []>
    : Op<ZAPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// BufferStackAllocation
//===----------------------------------------------------------------------===//

def ZAP_BufferStackAllocationOp : ZAPOp<"buffer.stack_allocation"> {
  let summary = "buffer scoped stack memory allocation operation";
  let description = [{
    The `zap.buffer.stack_allocation` operation allocates memory on the stack.
    The allocated memory is automatically released when control transfers back
    from the region of its closest surrounding operation. The amount of memory
    allocated is specified by the buffer type.

    The resulting buffer type must be concrete in both the size and dtype.
    That is, The buffer type cannot have `?` as either the size or dtype.

    Example:

    ```mlir
    %0 = zap.buffer.stack_allocation : !meta.buffer<4, f32>
    %1 = zap.buffer.stack_allocation : !meta.buffer<size, f32>
    %2 = zap.buffer.stack_allocation : !meta.buffer<size, type>
    // The following is not allowed
    %3 = pop.buffer.stack_allocation : !meta.buffer<?, f32>
    ```
  }];

  let results = (outs Res<ConcreteBuffer, "", [MemAlloc]>:$result);

  let assemblyFormat = "attr-dict `:` qualified(type($result))";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BufferLoadOp
//===----------------------------------------------------------------------===//

def ZAP_BufferLoadOp : ZAPOp<"buffer.load", [
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

  let assemblyFormat = [{
    $buffer `[` $position `]` attr-dict `:` qualified(type($buffer))
  }];
}

//===----------------------------------------------------------------------===//
// BufferStoreOp
//===----------------------------------------------------------------------===//

def ZAP_BufferStoreOp : ZAPOp<"buffer.store", [
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
    $value `,` $buffer `[` $position `]` attr-dict `:` qualified(type($buffer))
  }];
}


//===----------------------------------------------------------------------===//
// SIMDLoadOp
//===----------------------------------------------------------------------===//

def ZAP_SIMDLoadOp : ZAPOp<"simd.load"> {
  let summary = "simd load operation";
  let description = [{
    The `zap.simd.load` operation takes a buffer and an index position and
    extracts the `!meta.simd` value from the specified position. The
    position must be of type index.

    Example:

    ```mlir
    %1 = zap.buffer.load %0[%idx]: !meta.buffer<size, dtype>, !meta.simd<4, f32>
    ```
  }];

  let arguments = (ins Arg<ConcreteBuffer, "", [MemRead]>:$buffer,
                       Index:$position);
  let results = (outs SIMDType:$result);

  let assemblyFormat =  [{
    $buffer `[` $position `]` attr-dict `:`
        qualified(type($buffer)) `,` qualified(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BufferStoreOp
//===----------------------------------------------------------------------===//

def ZAP_SIMDStoreOp : ZAPOp<"simd.store"> {
  let summary = "buffer store operation";
  let description = [{
    The `zap.simd.store` operation takes a source value, a buffer, and an index
    position and copies the eleemnt of simd value into the buffer starting at
    the specified position. The position must be of type index.

    Example:

    ```mlir
    zap.simd.store %val, %buf[%idx]: !meta.simd<4, f32>, !meta.buffer<size, f32>
    ```
  }];

  let arguments = (ins Arg<ConcreteBuffer, "", [MemWrite]>:$buffer,
                       SIMDType:$value, Index:$position);

  let assemblyFormat = [{
    $value `,` $buffer `[` $position `]` attr-dict `:`
            qualified(type($value)) `,` qualified(type($buffer))
  }];
  let hasVerifier = 1;
}


#endif // ZAPOPS_TD
