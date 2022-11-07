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
    allocated is specified by the ndbuffer type. An optional `algin` attribute
    can be provided to specify the alignment of the allocation.

    The resulting ndbuffer type must be concrete in both the shape and dtype.
    That is, the ndbuffer type cannot have `?` as either the shape or dtype.

    Example:

    ```mlir
    %0 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[4], f32>
    %1 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[size0, size1], f32>
    %2 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[size], type>
    %2 = zap.ndbuffer.stack_allocation align 4 : !zap.ndbuffer<[size], type>

    // The following is not allowed:
    %3 = pop.ndbuffer.stack_allocation : !zap.ndbuffer<[?], f32>
    %4 = pop.ndbuffer.stack_allocation : !zap.ndbuffer<[4], ?>
    ```
  }];

  let arguments = (ins OptionalAttr<TypedAttrInterface>:$alignment);
  let results = (outs Res<ConcreteNDBuffer, "", [MemAlloc]>:$result);

  let assemblyFormat = [{
    custom<OptionalAlignmentParamValue>($alignment) `` attr-dict `:`
    qualified(type($result))
  }];
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

def ZAP_NDBufferLoadOp : ZAPOp<"ndbuffer.load"> {
  let summary = "ndbuffer load operation";
  let description = [{
    The `zap.ndbuffer.load` operation takes a ndbuffer and a list of index
    positions and extracts the `!pop.simd` value at that position. The
    positions must be of type index and their length must be of the same rank
    as the NDBuffer.

    Example:

    ```mlir
    %1 = zap.ndbuffer.load %0[%pos0] :
                      !zap.ndbuffer<[size], dtype>, !pop.simd<simd_width, dtype>
    %2 = zap.ndbuffer.load %1[%pos0, %pos1]:!zap.ndbuffer<[s0, s1], dtype>,
                                                 !pop.simd<simd_width, dtype>
    ```

    An optional `align` attribute can be specified to indicate the load
    alignment in bytes. If not specified, the alignment is assumed to be 1.

    ```mlir
    %1 = zap.ndbuffer.load %0[%pos0] align get_alignof(f32):
                      !zap.ndbuffer<[size], dtype>, !pop.simd<simd_width, dtype>
    %2 = zap.ndbuffer.load %1[%pos0, %pos1] align 64 :
                                            !zap.ndbuffer<[s0, s1], dtype>,
                                            !pop.simd<simd_width, dtype>
    ```
  }];

  let arguments = (ins Arg<NDBufferKnownDType, "", [MemRead]>:$nDBuffer,
                       Variadic<Index>:$positions,
                       OptionalAttr<TypedAttrInterface>:$alignment);
  let results = (outs POP_SIMDType:$result);

  let assemblyFormat = [{
    $nDBuffer `[` $positions `]` custom<OptionalAlignmentParamValue>($alignment)
    attr-dict `:` qualified(type($nDBuffer)) `,` qualified(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// NDBufferStoreOp
//===----------------------------------------------------------------------===//

def ZAP_NDBufferStoreOp : ZAPOp<"ndbuffer.store"> {
  let summary = "ndbuffer store operation";
  let description = [{
    The `zap.ndbuffer.store` operation takes a source simd value, a
    NDBuffer, and index positions and inserts the source value into the
    ndbuffer at the specified position. The positions must be of type index and
    their length must be of the same rank as the NDBuffer.

    Example:

    ```mlir
    %1 = zap.ndbuffer.store %val, %0[%pos0] : !pop.simd<simd_width, dtype>,
                                              !zap.ndbuffer<[size], dtype>
    %2 = zap.ndbuffer.store %val, %1[%pos0, %pos1] :
                                    !pop.simd<simd_width, dtype>,
                                    !zap.ndbuffer<[size0, size1], dtype>
    ```

    An optional `align` attribute can be specified to indicate the store
    alignment in bytes. If not specified, the alignment is assumed to be 1.

    ```mlir
    %1 = zap.ndbuffer.store %val, %0[%pos0] align 4 :
                                            !pop.simd<simd_width, dtype>,
                                            !zap.ndbuffer<[size], dtype>
    %2 = zap.ndbuffer.store %val, %1[%pos0, %pos1] align 1 :
                                   !pop.simd<simd_width, dtype>,
                                   !zap.ndbuffer<[size0, size1], dtype>
    ```
  }];

  let arguments = (ins Arg<NDBufferKnownDType, "", [MemWrite]>:$nDBuffer,
                       POP_SIMDType:$value,
                       Variadic<Index>:$positions,
                       OptionalAttr<TypedAttrInterface>:$alignment);

  let assemblyFormat = [{
    $value `,` $nDBuffer `[` $positions `]`
    custom<OptionalAlignmentParamValue>($alignment)
    attr-dict `:` qualified(type($value)) `,` qualified(type($nDBuffer))
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
    %1 = zap.ndbuffer.address %arg1 : !zap.ndbuffer<[?, 3], ?>
    ```
  }];

  let arguments = (ins ZAP_NDBufferType:$nDBuffer);
  let results = (outs POP_PointerType:$ptr);
  let assemblyFormat = "$nDBuffer attr-dict `:` qualified(type($nDBuffer))";
}

//===----------------------------------------------------------------------===//
// NDBufferBitCastOp
//===----------------------------------------------------------------------===//

def ZAP_NDBufferBitCastOp : ZAPOp<"ndbuffer.bitcast", [Pure,
    DeclareOpInterfaceMethods<CastOpInterface>,
    SameNDBufferRank<"input", "output">]> {
  let summary = "bitcasts an ndbuffer type";
  let description = [{
    The `zap.ndbuffer.bitcast` operation bitcasts an ndbuffer type to another
    ndbuffer type. It can convert between ndbuffers with static shape or
    dynamic shape. The rank of the source ndbuffer type and the destination
    ndbuffer type must match. When the input dtype is known, but output dtype
    is not, we propagate the input dtype to the output.

    Example:

    ```mlir
    // Bitcast a dynamic ndbuffer to a dynamic size ndbuffer.
    %0 = zap.ndbuffer.bitcast %a : !zap.ndbuffer<[?], f64> to
                                   !zap.ndbuffer<[?], f32>

    // Bitcast a static buffer to a dynamic size buffer.
    %1 = zap.ndbuffer.bitcast %b : !zap.ndbuffer<[1024], f32> to
                                   !zap.ndbuffer<[?], f32>

    // Bitcast an ndbuffer to a dynamic size ndbuffer.
    %2 = zap.ndbuffer.bitcast %c : !zap.ndbuffer<[32, 5], type> to
                                   !zap.ndbuffer<[?, 5], f32>

    // Bitcast across dtypes.
    %2 = zap.ndbuffer.bitcast %c : !zap.ndbuffer<[5], type1> to
                                   !zap.ndbuffer<[5], type2>
    ```

    The operation can bitcast between ndbuffer types of the same rank with
    different sizes or dtypes.

    Example:

    ```mlir
    // Bitcast an ndbuffer of shape 32x2 to an ndbuffer of shape 1x64.
    %0 = zap.ndbuffer.bitcast %a : !zap.ndbuffer<[32, 2], f32> to
                                   !zap.ndbuffer<[1, 64], f32>

    // Bitcast a ui32 ndbuffer to a si32 ndbuffer.
    %0 = zap.ndbuffer.bitcast %b : !zap.ndbuffer<[?], ui32> to
                                   !zap.ndbuffer<[?], si32>
    ```

    The operation can bitcast between ndbuffers of uknown type.

    ```mlir
    // Bitcast an ndbuffer of shape 32x2 but dynamic dtype to an ndbuffer
    // of the same shape but static dtype.
    %0 = zap.ndbuffer.bitcast %a : !zap.ndbuffer<[32, 2], ?> to
                                   !zap.ndbuffer<[32, 2], f32>

    // Bitcast an f32 ndbuffer to a generic ndbuffer.
    %0 = zap.ndbuffer.bitcast %b : !zap.ndbuffer<[?], f32> to
                                   !zap.ndbuffer<[?], ?>
    ```

    If the shape or dtype is concrete in the result type, it overwrites the
    shape or dtype in the input ndbuffer.
  }];

  let arguments = (ins ZAP_NDBufferType:$input);
  let results = (outs ZAP_NDBufferType:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `to` qualified(type($output))
  }];

  let hasFolder = 1;
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
    literal and only returns pointers to `!pop.array<N, scalar<si8>>`.

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

    Note: Unlike the `assert` operation in C, this operation is not fully
    removed when assertions are disabled. Therefore, production code should
    not use this operation. This operation is meant for debugging and
    verification purposes only.

    Example:

    ```mlir
    zap.debug_assert %cond, "a description of the condition"
    ```

    TODO: This Op should be removed in production builds.
  }];

  let arguments = (ins BoolSIMDSizeOne:$cond, StrAttr:$msg);

  let assemblyFormat = "$cond `,` $msg attr-dict `:` type($cond)";
}

#endif // ZAPOPS_TD
