//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ZAPOPS_TD
#define ZAPOPS_TD

include "KGEN/POPDialect/POPTypes.td"
include "KGEN/ZAPDialect/ZAPDialect.td"
include "KGEN/ZAPDialect/ZAPTypes.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ZAPOp
//===----------------------------------------------------------------------===//

/// Base class for ZAP dialect operations.
class ZAPOp<string mnemonic, list<Trait> traits = []>
    : Op<ZAPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// BufferConstructOp
//===----------------------------------------------------------------------===//

def ZAP_BufferConstructOp : ZAPOp<"buffer.construct", [NoSideEffect,
    AttrSizedOperandSegments, BufferPointerType<"result", "ptr">]> {
  let summary = "construct a buffer with a pointer";
  let description = [{
    The `zap.buffer.construct` operation constructs a buffer around a base
    pointer. Requires size and/or dtype operands as necessary, if they are
    dynamic, so not known by the return type.

    Example:

    ```mlir
    // Construct a buffer with a known size and type.
    %0 = zap.buffer.construct %ptr : !zap.buffer<4, f32>

    // Construct a buffer with a dynamic size.
    %1 = zap.buffer.construct %ptr[%size] : !zap.buffer<?, f32>

    // Construct a buffer with a dynamic dtype.
    %2 = zap.buffer.construct %ptr of %dtype : !zap.buffer<4, ?>

    // Construct a buffer with a dynamic size and dtype.
    %3 = zap.buffer.construct %ptr[%size] of %dtype : !zap.buffer<?, ?>
    ```
  }];

  let arguments = (ins POP_PointerType:$ptr, Optional<Index>:$size,
                       Optional<DTypeType>:$dType);
  let results = (outs ZAP_BufferType:$result);
  let assemblyFormat = [{
    $ptr (`[` $size^ `]`)? (`of` $dType^)? attr-dict `:`
    qualified(type($result))
  }];
  let hasVerifier = 1;

  let builders = [OpBuilder<(ins "Type":$type, "Value":$ptr)>];
}

//===----------------------------------------------------------------------===//
// BufferSizeOp
//===----------------------------------------------------------------------===//

def ZAP_BufferSizeOp : ZAPOp<"buffer.size", [NoSideEffect]> {
  let summary = "get the size of a buffer";
  let description = [{
    The `zap.buffer.size` operation returns the size of the given buffer as an
    SSA value (of type `index`) that may be used by other ops.

    Example:

    ```mlir
    %0 = zap.buffer.size %arg0 : !zap.buffer<3, f32>
    %1 = zap.buffer.size %arg1 : !zap.buffer<?, f32>
    ```
  }];

  let arguments = (ins ZAP_BufferType:$buffer);
  let results = (outs Index:$result);
  let assemblyFormat = "$buffer attr-dict `:` qualified(type($buffer))";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// BufferDTypeOp
//===----------------------------------------------------------------------===//

def ZAP_BufferDTypeOp : ZAPOp<"buffer.dtype", [NoSideEffect]> {
  let summary = "get the dtype of a buffer";
  let description = [{
    The `zap.buffer.dtype` operation returns the dtype of the given buffer as an
    SSA value (of type `!kgen.dtype`) that may be used by other ops.

    Example:

    ```mlir
    %0 = zap.buffer.dtype %arg0 : !zap.buffer<3, f32>
    %1 = zap.buffer.dtype %arg0 : !zap.buffer<3, ?>
    ```
  }];

  let arguments = (ins ZAP_BufferType:$buffer);
  let results = (outs DTypeType:$dtype);
  let assemblyFormat = "$buffer attr-dict `:` qualified(type($buffer))";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// BufferAddressOp
//===----------------------------------------------------------------------===//

def ZAP_BufferAddressOp : ZAPOp<"buffer.address", [NoSideEffect,
    BufferPointerType<"buffer", "ptr">]> {
  let summary = "get the starting address of a buffer";
  let description = [{
    The `zap.buffer.address` operation returns the address of the given buffer
    as a pointer to a scalar of the buffer dtype.

    Example:

    ```mlir
    %0 = zap.buffer.address %arg0 : !zap.buffer<3, f32>
    %1 = zap.buffer.address %arg1 : !zap.buffer<3, ?>
    ```
  }];

  let arguments = (ins ZAP_BufferType:$buffer);
  let results = (outs POP_PointerType:$ptr);
  let assemblyFormat = "$buffer attr-dict `:` qualified(type($buffer))";
}

//===----------------------------------------------------------------------===//
// BufferConvertOp
//===----------------------------------------------------------------------===//

def ZAP_BufferConvertOp : ZAPOp<"buffer.bitcast", [NoSideEffect,
    DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "cast a buffer type";
  let description = [{
    The `zap.buffer.convert` operation converts a buffer type to another buffer
    type. It can convert between buffers with static sizes and dynamic sizes and
    between those of static dtypes and dynamic dtypes.

    Example:

    ```mlir
    // Convert a dynamic buffer to a dynamic size buffer.
    %0 = zap.buffer.bitcast %a : !zap.buffer<?, ?> to !zap.buffer<?, f32>

    // Convert a static buffer to a dynamic size buffer.
    %1 = zap.buffer.bitcast %b : !zap.buffer<1024, f32> to !zap.buffer<?, f32>

    // Convert a dynamic dtype buffer to a dynamic size buffer.
    %2 = zap.buffer.bitcast %c : !zap.buffer<32, ?> to !zap.buffer<?, f32>
    ```

    The operation can bitcast between buffer types of the same arity with
    different sizes or dtypes.

    Example:

    ```mlir
    // Convert a size 32 buffer to a size 64 buffer.
    %0 = zap.buffer.bitcast %a : !zap.buffer<32, ?> to !zap.buffer<64, ?>

    // Convert a ui32 buffer to a si32 buffer.
    %0 = zap.buffer.bitcast %b : !zap.buffer<?, ui32> to !zap.buffer<?, si32>
    ```

    If the size or dtype is concrete in the result type, it overwrites the
    size or dtype in the input buffer.
  }];

  let arguments = (ins ZAP_BufferType:$input);
  let results = (outs ZAP_BufferType:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `to` qualified(type($output))
  }];

  let hasFolder = 1;
}

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
    %0 = zap.buffer.stack_allocation : !zap.buffer<4, f32>
    %1 = zap.buffer.stack_allocation : !zap.buffer<size, f32>
    %2 = zap.buffer.stack_allocation : !zap.buffer<size, type>
    // The following is not allowed
    %3 = pop.buffer.stack_allocation : !zap.buffer<?, f32>
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

def ZAP_BufferLoadOp : ZAPOp<"buffer.load",
    [BufferElementType<"buffer", "result">]> {
  let summary = "buffer load operation";
  let description = [{
    The `buffer.load` operation takes a buffer and an index position and
    extracts the `!pop.scalar` value at that position. The position must be of
    type index.

    Example:

    ```mlir
    %1 = zap.buffer.load %0[%position]: !zap.buffer<size, dtype>
    ```
  }];

  let arguments = (ins Arg<BufferKnownDType, "", [MemRead]>:$buffer,
                       Index:$position);
  let results = (outs POP_ScalarType:$result);

  let assemblyFormat = [{
    $buffer `[` $position `]` attr-dict `:` qualified(type($buffer))
  }];
}

//===----------------------------------------------------------------------===//
// BufferStoreOp
//===----------------------------------------------------------------------===//

def ZAP_BufferStoreOp : ZAPOp<"buffer.store",
    [BufferElementType<"buffer", "value">]> {
  let summary = "buffer store operation";
  let description = [{
    The `buffer.store.` operation takes a source value, a buffer, and an index
    position and inserts the source value into the buffer at that position. The
    position must be of type index.

    Example:

    ```mlir
    zap.buffer.store %val, %buf[%position]: !zap.buffer<size, dtype>
    ```
  }];

  let arguments = (ins Arg<BufferKnownDType, "", [MemWrite]>:$buffer,
                       POP_ScalarType:$value, Index:$position);

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
    extracts the `!pop.simd` value from the specified position. The
    position must be of type index.

    Example:

    ```mlir
    %1 = zap.buffer.load %0[%idx]: !zap.buffer<size, dtype>, !pop.simd<4, f32>
    ```
  }];

  let arguments = (ins Arg<BufferKnownDType, "", [MemRead]>:$buffer,
                       Index:$position);
  let results = (outs POP_SIMDType:$result);

  let assemblyFormat =  [{
    $buffer `[` $position `]` attr-dict `:`
        qualified(type($buffer)) `,` qualified(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SIMDStoreOp
//===----------------------------------------------------------------------===//

def ZAP_SIMDStoreOp : ZAPOp<"simd.store"> {
  let summary = "buffer store operation";
  let description = [{
    The `zap.simd.store` operation takes a source value, a buffer, and an index
    position and copies the eleemnt of simd value into the buffer starting at
    the specified position. The position must be of type index.

    Example:

    ```mlir
    zap.simd.store %val, %buf[%idx]: !pop.simd<4, f32>, !zap.buffer<size, f32>
    ```
  }];

  let arguments = (ins Arg<BufferKnownDType, "", [MemWrite]>:$buffer,
                       POP_SIMDType:$value, Index:$position);

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
    zap.print "foo %d / %f bar"(%a, %b) : !pop.scalar<type>,
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
