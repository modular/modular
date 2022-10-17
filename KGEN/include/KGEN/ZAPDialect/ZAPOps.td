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
include "Support/MDialect/MAttrs.td"
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

def ZAP_BufferConstructOp : ZAPOp<"buffer.construct", [Pure,
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

def ZAP_BufferSizeOp : ZAPOp<"buffer.size", [Pure]> {
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

def ZAP_BufferDTypeOp : ZAPOp<"buffer.dtype", [Pure]> {
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

def ZAP_BufferAddressOp : ZAPOp<"buffer.address", [Pure,
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

def ZAP_BufferConvertOp : ZAPOp<"buffer.bitcast", [Pure,
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

def ZAP_BufferConstantOp : ZAPOp<"buffer.constant"> {
  let summary = "fill a buffer with constant values";
  let description = [{
    The `zap.buffer.constant` operation returns a global constant buffer with
    the specified constant values. The result buffer always has a statically
    known size. The memory semantics of the operation are similar to
    `static const`. Writing to the buffer is undefined behaviour.

    Example:

    ```mlir
    %0 = zap.buffer.constant(
        #M.dense_array<10.0, 12.0, -2.0> : !M.array<3xf32>) : f32
    %1 = zap.buffer.constant(#M.dense_array<2, 3> : !M.array<2xui8>) : type
    ```
  }];

  let arguments = (ins M_ArrayElementsAttr:$values);
  let results = (outs Res<ConcreteBuffer, "", [MemAlloc, MemWrite]>:$result);

  let assemblyFormat = [{
    `(` qualified($values) `)` attr-dict `:`
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

def ZAP_BufferSIMDLoadOp : ZAPOp<"buffer.simd_load"> {
  let summary = "simd load operation";
  let description = [{
    The `zap.buffer.simd_load` operation takes a buffer and an index position
    and extracts the `!pop.simd` value from the specified position. The
    position must be of type index.

    Example:

    ```mlir
    %val = zap.buffer.simd_load %buf[%idx]: !zap.buffer<size,dtype>, !pop.simd<4,f32>
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

def ZAP_BufferSIMDStoreOp : ZAPOp<"buffer.simd_store"> {
  let summary = "buffer store operation";
  let description = [{
    The `zap.buffer.simd_store` operation takes a source value, a buffer, and
    an index position and copies the eleemnt of simd value into the buffer
    starting at the specified position. The position must be of type index.

    Example:

    ```mlir
    zap.buffer.simd_store %val, %buf[%idx]: !pop.simd<4, f32>, !zap.buffer<size, f32>
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
// NDBufferConstructOp
//===----------------------------------------------------------------------===//

def ZAP_NDBufferConstructOp : ZAPOp<"ndbuffer.construct", [Pure,
    AttrSizedOperandSegments, NDBufferPointerType<"result", "ptr">]> {
  let summary = "construct a type with a pointer";
  let description = [{
    The `zap.ndbuffer.construct` operation constructs a ndbuffer using a raw
    pointer. Requires size and/or dtype operands as necessary, if they are
    dynamic, so not known by the return type. The order of the operands bind
    to the order of unknown sizes in the return type.

    Example:

    ```mlir
    // Construct a ndbuffer with a known size and type.
    %0 = zap.ndbuffer.construct %ptr : !zap.ndbuffer<[4], f32>

    // Construct a 2-dimensional ndbuffer with a dynamic size.
    %1 = zap.ndbuffer.construct %ptr[%s1, %s2] : !zap.ndbuffer<[?, ?], f32>

    // Construct a 2-dimensional ndbuffer with the last dimension being dynamic.
    %2 = zap.ndbuffer.construct %ptr[%s1] : !zap.ndbuffer<[42, ?], f32>

    // Construct a ndbuffer with a dynamic dtype.
    %3 = zap.ndbuffer.construct %ptr of %dtype : !zap.ndbuffer<[4], ?>

    // Construct a buffer with a dynamic size and dtype.
    %4 = zap.ndbuffer.construct %ptr[%size] of %dtype : !zap.ndbuffer<[?], ?>
    ```
  }];

  let arguments = (ins POP_PointerType:$ptr, Variadic<Index>:$shape,
                       Optional<DTypeType>:$dType);
  let results = (outs ZAP_NDBufferType:$result);
  let assemblyFormat = [{
    $ptr (`[` $shape^ `]`)? (`of` $dType^)? attr-dict `:`
    qualified(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// NDBufferStackAllocation
//===----------------------------------------------------------------------===//

def ZAP_NDBufferStackAllocationOp : ZAPOp<"ndbuffer.stack_allocation"> {
  let summary = "ndbuffer scoped stack memory allocation operation";
  let description = [{
    The `zap.ndbuffer.stack_allocation` operation allocates memory on the stack.
    The allocated memory is automatically released when control transfers back
    from the region of its closest surrounding operation. The amount of memory
    allocated is specified by the ndbuffer type.

    The resulting ndbuffer type must be concrete in both the shape and dtype.
    That is, the ndbuffer type cannot have `?` as either the shape or dtype.

    Example:

    ```mlir
    %0 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[4], f32>
    %1 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[size0, size1], f32>
    %2 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[size], type>

    // The following is not allowed:
    %3 = pop.ndbuffer.stack_allocation : !zap.ndbuffer<[?], f32>
    %4 = pop.ndbuffer.stack_allocation : !zap.ndbuffer<[4], ?>
    ```
  }];

  let results = (outs Res<ConcreteNDBuffer, "", [MemAlloc]>:$result);

  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// NDBufferDimOp
//===----------------------------------------------------------------------===//

def ZAP_NDBufferDimOp : ZAPOp<"ndbuffer.dim", [Pure]> {
  let summary = "get the dimension of the ndbuffer at specified index";
  let description = [{
    The `zap.ndbuffer.dim` operation returns the size of the given ndbuffer as
    an SSA value (of type `index`) at the specified dimension.

    Example:

    ```mlir
    %0 = zap.ndbuffer.dim %arg0[%idx] : !zap.ndbuffer<[3], f32>
    %1 = zap.ndbuffer.dim %arg1[%idx] : !zap.ndbuffer<[4, ?], f32>
    ```
  }];

  let arguments = (ins ZAP_NDBufferType:$nDBuffer,
                       ConfinedAttr<IndexAttr, [IntNonNegative]>:$index);
  let results = (outs Index:$result);
  let assemblyFormat = [{
    $nDBuffer `[` $index `]` attr-dict `:` qualified(type($nDBuffer))
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// NDBufferSizeOp
//===----------------------------------------------------------------------===//

def ZAP_NDBufferSizeOp : ZAPOp<"ndbuffer.size", [Pure]> {
  let summary = "get the total number of elements in the NDBuffer";
  let description = [{
    The `zap.ndbuffer.size` operation returns the total number of elements in
    the given ndbuffer as an SSA value (of type `index`).

    Example:

    ```mlir
    %0 = zap.ndbuffer.size %arg0 : !zap.ndbuffer<[3], f32>
    %1 = zap.ndbuffer.size %arg1 : !zap.ndbuffer<[?, 4, ?], ?>
    ```
  }];

  let arguments = (ins ZAP_NDBufferType:$nDBuffer);
  let results = (outs Index:$result);
  let assemblyFormat = "$nDBuffer attr-dict `:` qualified(type($nDBuffer))";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// NDBufferLoadOp
//===----------------------------------------------------------------------===//

def ZAP_NDBufferLoadOp : ZAPOp<"ndbuffer.load",
    [NDBufferElementType<"nDBuffer", "result">]> {
  let summary = "ndbuffer load operation";
  let description = [{
    The `zap.ndbuffer.load` operation takes a ndbuffer and a list of index
    positions and extracts the `!pop.scalar` value at that position. The
    positions must be of type index and their length must be of the same rank
    as the NDBuffer.

    Example:

    ```mlir
    %1 = zap.ndbuffer.load %0[%pos0] : !zap.ndbuffer<[size], dtype>
    %2 = zap.ndbuffer.load %1[%pos0, %pos1]:!zap.ndbuffer<[size0, size1], dtype>
    ```
  }];

  let arguments = (ins Arg<NDBufferKnownDType, "", [MemRead]>:$nDBuffer,
                       Variadic<Index>:$positions);
  let results = (outs POP_ScalarType:$result);

  let assemblyFormat = [{
    $nDBuffer `[` $positions `]` attr-dict `:` qualified(type($nDBuffer))
  }];

  let hasVerifier = 1;
}
//===----------------------------------------------------------------------===//

// NDBufferStoreOp
//===----------------------------------------------------------------------===//

def ZAP_NDBufferStoreOp : ZAPOp<"ndbuffer.store",
    [NDBufferElementType<"nDBuffer", "value">]> {
  let summary = "ndbuffer store operation";
  let description = [{
    The `zap.ndbuffer.store` operation takes a source value, a buffer, and index
    positions and inserts the source value into the ndbuffer at the specified
    position. The positions must be of type index and their length must be of
    the same rank as the NDBuffer.

    Example:

    ```mlir
    %1 = zap.ndbuffer.store %val, %0[%pos0] : !zap.ndbuffer<[size], dtype>
    %2 = zap.ndbuffer.store %val, %1[%pos0, %pos1] :
                                      !zap.ndbuffer<[size0, size1], dtype>
    ```
  }];

  let arguments = (ins Arg<NDBufferKnownDType, "", [MemWrite]>:$nDBuffer,
                       POP_ScalarType:$value, Variadic<Index>:$positions);

  let assemblyFormat = [{
    $value `,` $nDBuffer `[` $positions `]` attr-dict `:` qualified(type($nDBuffer))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// NDBufferSIMDLoadOp
//===----------------------------------------------------------------------===//

def ZAP_NDBufferSIMDLoadOp : ZAPOp<"ndbuffer.simd_load"> {
  let summary = "ndbuffer load operation";
  let description = [{
    The `zap.ndbuffer.simd_load` operation takes a ndbuffer and a list of index
    positions and extracts the `!pop.simd` value at that position. The
    positions must be of type index and their length must be of the same rank
    as the NDBuffer.

    Example:

    ```mlir
    %1 = zap.ndbuffer.simd_load %0[%pos0] :
                      !zap.ndbuffer<[size], dtype>, !pop.simd<simd_width, dtype>
    %2 = zap.ndbuffer.simd_load %1[%pos0, %pos1]:!zap.ndbuffer<[s0, s1], dtype>,
                                                 !pop.simd<simd_width, dtype>
    ```
  }];

  let arguments = (ins Arg<NDBufferKnownDType, "", [MemRead]>:$nDBuffer,
                       Variadic<Index>:$positions);
  let results = (outs POP_SIMDType:$result);

  let assemblyFormat = [{
    $nDBuffer `[` $positions `]` attr-dict `:`
      qualified(type($nDBuffer)) `,` qualified(type($result))
  }];

  let hasVerifier = 1;
}
//===----------------------------------------------------------------------===//

// NDBufferSIMDStoreOp
//===----------------------------------------------------------------------===//

def ZAP_NDBufferSIMDStoreOp : ZAPOp<"ndbuffer.simd_store"> {
  let summary = "ndbuffer store operation";
  let description = [{
    The `zap.ndbuffer.simd_store` operation takes a source simd value, a
    NDBuffer, and index positions and inserts the source value into the
    ndbuffer at the specified position. The positions must be of type index and
    their length must be of the same rank as the NDBuffer.

    Example:

    ```mlir
    %1 = zap.ndbuffer.simd_store %val, %0[%pos0] : !pop.simd<simd_width, dtype>,
                                                 !zap.ndbuffer<[size], dtype>
    %2 = zap.ndbuffer.simd_store %val, %1[%pos0, %pos1] :
                                        !pop.simd<simd_width, dtype>,
                                        !zap.ndbuffer<[size0, size1], dtype>
    ```
  }];

  let arguments = (ins Arg<NDBufferKnownDType, "", [MemWrite]>:$nDBuffer,
                       POP_SIMDType:$value, Variadic<Index>:$positions);

  let assemblyFormat = [{
    $value `,` $nDBuffer `[` $positions `]` attr-dict `:`
      qualified(type($value)) `,` qualified(type($nDBuffer))
  }];

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// NDBufferDTypeOp
//===----------------------------------------------------------------------===//

def ZAP_NDBufferDTypeOp : ZAPOp<"ndbuffer.dtype", [Pure]> {
  let summary = "get the dtype of the NDBuffer";
  let description = [{
    The `zap.ndbuffer.dtype` operation returns the dtype of the given ndbuffer
    as an SSA value (of type `!kgen.dtype`) that may be used by other ops.

    Example:

    ```mlir
    %0 = zap.ndbuffer.dtype %arg0 : !zap.ndbuffer<[3], ?>
    %1 = zap.ndbuffer.dtype %arg1 : !zap.ndbuffer<[4, ?], f32>
    ```
  }];

  let arguments = (ins ZAP_NDBufferType:$nDBuffer);
  let results = (outs DTypeType:$result);
  let assemblyFormat = "$nDBuffer attr-dict `:` qualified(type($nDBuffer))";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// NDBufferRankOp
//===----------------------------------------------------------------------===//

def ZAP_NDBufferRankOp : ZAPOp<"ndbuffer.rank", [Pure]> {
  let summary = "get the rank of the NDBuffer";
  let description = [{
    The `zap.ndbuffer.rank` operation returns the rank of the given ndbuffer as
    an SSA value (of type `index`) that may be used by other ops.

    Example:

    ```mlir
    %0 = zap.ndbuffer.rank %arg0 : !zap.ndbuffer<[3], f32>
    %1 = zap.ndbuffer.rank %arg1 : !zap.ndbuffer<[4, ?], f32>
    ```
  }];

  let arguments = (ins ZAP_NDBufferType:$nDBuffer);
  let results = (outs Index:$result);
  let assemblyFormat = "$nDBuffer attr-dict `:` qualified(type($nDBuffer))";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// NDBufferAddressOp
//===----------------------------------------------------------------------===//

def ZAP_NDBufferAddressOp : ZAPOp<"ndbuffer.address", [Pure,
    NDBufferPointerType<"nDBuffer", "ptr">]> {
  let summary = "get the address of the NDBuffer";
  let description = [{
    The `zap.ndbuffer.address` operation returns the address of the given
    NDBuffer as a pointer to a scalar of the ndbuffer dtype. The pointer is to
    the start of the underlying ndbuffer data.

    Example:

    ```mlir
    %0 = zap.ndbuffer.address %arg0 : !zap.ndbuffer<[3], f32>
    %1 = zap.ndbuffer.address %arg1 : !zap.buffer<[?, 3], ?>
    ```
  }];

  let arguments = (ins ZAP_NDBufferType:$nDBuffer);
  let results = (outs POP_PointerType:$ptr);
  let assemblyFormat = "$nDBuffer attr-dict `:` qualified(type($nDBuffer))";
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def ZAP_PrintOp : ZAPOp<"print"> {
  let summary = "print format operation";
  let description = [{
    Print a series of values formatted according to the provided string. This
    function is meant for debugging. The format string is the same as `printf`
    and may not be supported on all targets.

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

//===----------------------------------------------------------------------===//
// GlobalStringOp
//===----------------------------------------------------------------------===//

def KnownSizeSI8Array : ConfinedType<ArrayKnownSize,
    [CPred<"isSI8Array($_self)">], "array of scalar `si8`">;

def POP_GlobalStringOp : ZAPOp<"global_string", [Pure]> {
  let summary = "create a global string";
  let description = [{
    The `zap.global_string` operation creates a global string value and returns
    a pointer to it. The returned pointer is read-only. Writing to the pointer
    is undefined behaviour. The string is NOT null-terminated.

    This operation is similar to `pop.global_constant` except it uses a string
    literal and only returns pointers to `!pop.array<N, !pop.scalar<si8>>`.

    Example:

    ```mlir
    // Create a string.
    %1 = zap.global_string "foo"[3]

    // Create a null-terminated string.
    %1 = zap.global_string "hello world!!\00"[14]
    ```
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs PointerOf<KnownSizeSI8Array>:$result);
  let assemblyFormat = [{
    $value `[` custom<StringSizeArray>(type($result)) `]` attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DebugAssertOp
//===----------------------------------------------------------------------===//

def ZAP_DebugAssertOp : ZAPOp<"debug_assert"> {
  let summary = "asserts that the boolean value is true";
  let description = [{
    Asserts that the boolean value is true. If the value is false, the
    optional message is printed to `stderr` and the program aborts.

    Note: Unlike the `assert` operation in C, this operation is fully
    removed if assertions are disabled. Therefore, production code should
    not use this operation. This operation is meant for debugging and
    verification purposes only.

    Example:

    ```mlir
    zap.debug_assert %cond, "a description of the condition"
    ```

    TODO: This Op should be removed in production builds.
  }];

  let arguments = (ins BoolScalar:$cond, StrAttr:$msg);

  let assemblyFormat = "$cond `,` $msg attr-dict `:` type($cond)";
}

#endif // ZAPOPS_TD
