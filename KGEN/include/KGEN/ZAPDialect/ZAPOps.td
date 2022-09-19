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
// BufferConstant
//===----------------------------------------------------------------------===//

def IntOrFPElementsAttr
  : ElementsAttrBase<CPred<"$_self.isa<mlir::DenseIntOrFPElementsAttr>()">,
                     "and int or float elements attribute"> {
  let storageType = "mlir::DenseIntOrFPElementsAttr";
  let returnType = "mlir::DenseIntOrFPElementsAttr";
  let convertFromStorage = "$_self";
}

def ZAP_BufferConstantOp : ZAPOp<"buffer.constant"> {
  let summary = "fill a buffer with constant values";
  let description = [{
    The `zap.buffer.constant` operation returns a global constant buffer with
    the specified constant values. The result buffer always has a statically
    known size. The memory semantics of the operation are similar to
    `static const`. Writing to the buffer is undefined behaviour.

    Example:

    ```mlir
    %0 = zap.buffer.constant(dense<[10.0, 12.0, -2.0]> : tensor<3xf32>) : f32
    %1 = zap.buffer.constant(dense<[2, 3]> : tensor<2xui8>) : type
    ```
  }];

  let arguments = (ins IntOrFPElementsAttr:$values);
  let results = (outs Res<ConcreteBuffer, "", [MemAlloc, MemWrite]>:$result);

  let assemblyFormat = [{
    `(` $values `)` attr-dict `:`
    custom<ConstantBufferDType>(ref($values), type($result))
  }];

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

  let arguments = (ins Arg<BufferKnownDType, "", [MemRead]>:$buffer,
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

  let arguments = (ins Arg<BufferKnownDType, "", [MemWrite]>:$buffer,
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

  let arguments = (ins Arg<BufferKnownDType, "", [MemRead]>:$buffer,
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

  let arguments = (ins Arg<BufferKnownDType, "", [MemWrite]>:$buffer,
                       SIMDType:$value, Index:$position);

  let assemblyFormat = [{
    $value `,` $buffer `[` $position `]` attr-dict `:`
            qualified(type($value)) `,` qualified(type($buffer))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def ZAP_PrintOp : ZAPOp<"print"> {
  let summary = "print format operation";
  let description = [{
    Print a series of values formatted according to the provided string.
    This function is meant for debugging. The format string is the same as
    `printf` and may not be supported on all targets.

    Example:

    ```mlir
    zap.print "foo %d / %f bar"(%a, %b) : !meta.scalar<type>,
                                          !kgen.paramref<type>
    ```

    TODO: This op is target-specific and should be modelled as such.
  }];

  let arguments = (ins StrAttr:$fmt, Variadic<AnyType>:$operands);

  let assemblyFormat = [{
    $fmt `(` $operands `)` attr-dict (`:` type($operands)^)?
  }];
}

#endif // ZAPOPS_TD
