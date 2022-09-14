//===- MetaOps.td ---------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the Meta dialect.
//
//===----------------------------------------------------------------------===//

include "KGEN/MetaDialect/MetaDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operation in this dialect.
class MetaOp<string mnemonic, list<Trait> traits = []> :
    Op<MetaDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

class RebindOp<string mnemonic, Type type> : MetaOp<mnemonic, [NoSideEffect]> {
  let arguments = (ins type:$input);
  let results = (outs type:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `to` qualified(type($output))
  }];
  let hasVerifier = true;
  let hasFolder = true;
}

//===----------------------------------------------------------------------===//
// ScalarRebindOp
//===----------------------------------------------------------------------===//

def ScalarRebindOp : RebindOp<"scalar.rebind", ScalarType> {
  let summary = "rebind a scalar type";
  let description = [{
    This operation rebinds the dtype of a parameterized scalar type. The dtypes
    of scalar types can be rebound between parameter domains and concrete
    values, but rebinding a concrete dtype to a different concrete type is
    invalid.

    Example:

    ```mlir
    // Rebind dtype from one domain to another.
    %0 = meta.scalar.rebind %a : !meta.scalar<type1> to !meta.scalar<type2>

    // Rebind dtype from a concrete dtype to a parameterized dtype.
    %1 = meta.scalsr.rebind %b : !meta.scalar<f32> to !meta.scalar<type>

    // ERROR: Cannot rebind between different concrete dtypes.
    %2 = meta.scalar.rebind %c : !meta.scalar<ui32> to !meta.scalar<si32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SIMDRebindOp
//===----------------------------------------------------------------------===//

def SIMDRebindOp : RebindOp<"simd.rebind", SIMDType> {
  let description = "rebind a SIMD type";
  let description = [{
    This operation rebinds the size and dtype of a parameterized SIMD type. The
    sizes and dtypes of SIMD types can be rebound between parameter domains and
    concrete values, but rebinding a concrete size or dtype to a different
    concrete size or dtype is invalid.

    Example:

    ```mlir
    // Rebind size from one domain to another.
    %0 = meta.simd.rebind %a : !meta.simd<x, f32> to !meta.simd<add(y, 2), f32>

    // Bind a parameterized size to a concrete size.
    %1 = meta.simd.rebind %b : !meta.simd<size, type> to !meta.simd<4, type>

    // Unbind dtype from a concrete dtype to a parameterized dtype.
    %2 = meta.simd.rebind %c : !meta.simd<size, f32> to !meta.simd<size, type>

    // ERROR: Cannot rebind between different concrete sizes.
    %3 = meta.simd.rebind %d : !meta.simd<4, type> to !meta.simd<8, type>

    // ERROR: Cannot rebind between different concrete dtypes.
    %4 = meta.simd.rebind %e : !meta.simd<size, ui32> to !meta.simd<size, si32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// PointerRebindOp
//===----------------------------------------------------------------------===//

def PointerRebindOp : RebindOp<"pointer.rebind", PointerType> {
  let description = "rebind a pointer type";
  let description = [{
    This operation rebinds the dtype of a parameterized pointer type. The dtype
    can be rebound between parameter domains, concrete values, and unknown
    values, but rebinding a concrete dtype to a different concrete dtype is
    invalid.

    Example:

    ```mlir
    // Rebind dtype from one domain to another.
    %0 = meta.pointer.rebind %a : !meta.pointer<!meta.scalar<type1>> to
                                  !meta.pointer<!meta.scalar<type2>>

    // Bind a parameterized element type to a parameterized element type.
    %1 = meta.pointer.rebind %b : !meta.pointer<elementType> to
                                  !meta.pointer<!meta.scalar<type>>

    // Unbind a concrete element type to an unknown element type.
    %2 = meta.pointer.rebind %c : !meta.pointer<!meta.scalar<si32>> to
                                  !meta.pointer<?>

    // ERROR: Cannot rebind between different concrete types.
    %3 = meta.simd.rebind %d : !meta.pointer<!meta.scalar<f32>> to
                               !meta.pointer<!meta.simd<2, si32>>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// BufferConstructOp
//===----------------------------------------------------------------------===//

def BufferConstructOp : MetaOp<"buffer.construct", [
    NoSideEffect, AttrSizedOperandSegments,
    PointerOfSameDType<"result", "ptr">]> {
  let summary = "construct a buffer with a pointer";
  let description = [{
    Construct a buffer around a base pointer. Requires size and/or dtype
    operands as necessary, if they are unknown in the return type.

    Example:

    ```mlir
    // Construct a buffer with a known size and type.
    %0 = meta.buffer.construct %ptr : !meta.buffer<4, f32>

    // Construct a buffer with an unknown size.
    %1 = meta.buffer.construct %ptr[%size] : !meta.buffer<?, f32>

    // Construct a buffer with an unknown dtype.
    %2 = meta.buffer.construct %ptr of %dtype : !meta.buffer<4, ?>

    // Construct a buffer with an unknown size and dtype.
    %3 = meta.buffer.construct %ptr[%size] of %dtype : !meta.buffer<?, ?>
    ```

    A buffer with statically-known sizes and dtypes can be constructed with size
    or dtype operands, but they will be ignored.

    ```mlir
    %4 = meta.buffer.construct %ptr[%size] of %dtype : !meta.buffer<4, f32>
    ```
  }];

  let arguments = (ins PointerType:$ptr, Optional<Index>:$size,
                       Optional<DTypeType>:$dType);
  let results = (outs BufferType:$result);
  let assemblyFormat = [{
    $ptr (`[` $size^ `]`)? (`of` $dType^)? attr-dict `:`
    qualified(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BufferSizeOp
//===----------------------------------------------------------------------===//

def BufferSizeOp : MetaOp<"buffer.size", [NoSideEffect]> {
  let summary = "get the size of a buffer";
  let description = [{
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

//===----------------------------------------------------------------------===//
// BufferDTypeOp
//===----------------------------------------------------------------------===//

def BufferDTypeOp : MetaOp<"buffer.dtype", [NoSideEffect]> {
  let summary = "get the dtype of a buffer";
  let description = [{
    Return the dtype of a meta.buffer as an SSA value (of type `dtype`) that may
    be used by other ops.

    Example:

    ```mlir
    %0 = meta.buffer.dtype %arg0 : meta.buffer<3xf32>
    ```
  }];

  let arguments = (ins BufferType:$value);
  let results = (outs DTypeType:$dtype);
  let assemblyFormat = "$value attr-dict `:` qualified(type($value))";
  let hasFolder = true;
}

//===----------------------------------------------------------------------===//
// BufferAddressOp
//===----------------------------------------------------------------------===//

def BufferAddressOp : MetaOp<"buffer.address", [NoSideEffect,
                      DeclareOpInterfaceMethods<InferTypeOpInterface,
                                                ["inferReturnTypes"]>]> {
  let summary = "get the starting address of a buffer";
  let description = [{
    Return the address of a meta.buffer as an SSA value (of type !meta.pointer)
    that may be used by other ops.

    Example:

    ```mlir
    %0 = meta.buffer.address %arg0 : meta.buffer<3xf32>
    ```
  }];

  let arguments = (ins BufferType:$value);
  let results = (outs PointerType:$pointer);
  let assemblyFormat = "$value attr-dict `:` qualified(type($value))";
}

//===----------------------------------------------------------------------===//
// BufferRebindOp
//===----------------------------------------------------------------------===//

def BufferRebindOp : RebindOp<"buffer.rebind", BufferType> {
  let summary = "rebind a buffer type";
  let description = [{
    Rebind a parameterized buffer type to another buffer type. This operation
    is used to convert between buffer types that are potentially dynamically
    equal by specifying or de-specifying the size or dtype of the buffer.

    Example:

    ```mlir
    // Rebind an unknown dtype to a concrete dtype.
    %0 = meta.buffer.rebind %a : !meta.buffer<?, ?> to !meta.buffer<?, f32>

    // Rebind a concrete size to an unknown size.
    %1 = meta.buffer.rebind %b : !meta.buffer<1024, f32> to !meta.buffer<?, f32>

    // Rebind parameterized a size and dtype to a concrete size and dtype.
    %2 = meta.buffer.rebind %c : !meta.buffer<size, type> to !meta.buffer<1024, ui32>

    // Rebind a parameterized size to an unknown size.
    %3 = meta.buffer.rebind %d : !meta.buffer<size, f32> to !meta.buffer<?, f32>
    ```

    Rebinding between different concrete sizes or dtypes is not invalid:

    ```
    // ERROR: Cannot rebind different concrete sizes.
    %0 = meta.buffer.rebind %a : !meta.buffer<32, ?> to !meta.buffer<64, ?>

    // ERROR: Cannot rebind different concrete dtypes.
    %0 = meta.buffer.rebind %b : !meta.buffer<?, ui32> to !meta.buffer<?, si32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MetaCastToBuiltinOp
//===----------------------------------------------------------------------===//

def MetaCastToBuiltinOp : MetaOp<"cast_to_builtin", [NoSideEffect]> {
  let summary = "cast a meta type to a builtin type";
  let description = [{
    Cast the input scalar / simd to the corresponding MLIR standard type.

    Example:
    ```mlir
    %0 = meta.cast_to_builtin %scalar: !meta.scalar<f32> to f32
    %1 = meta.cast_to_builtin %simd : !meta.simd<4, f32> to vector<4xf32>
    ```
  }];

  let arguments = (ins AnyType:$in);
  let results = (outs AnyType:$out);

  let assemblyFormat = "$in `:` type($in) `to` type($out) attr-dict";
  let hasVerifier = true;
  let hasFolder = true;
}

//===----------------------------------------------------------------------===//
// MetaCastFromBuiltinOp
//===----------------------------------------------------------------------===//

def MetaCastFromBuiltinOp : MetaOp<"cast_from_builtin", [NoSideEffect]> {
  let summary = "cast a builtin type to a meta type";
  let description = [{
    Cast the input standard MLIR type to the corresponding !meta type.

    Example:
    ```mlir
    %0 = meta.cast_from_builtin %scalar: f32 to !meta.scalar<f32>
    %1 = meta.cast_from_builtin %simd : vector<4xf32> to !meta.simd<4, f32>
    ```
  }];

  let arguments = (ins AnyType:$in);
  let results = (outs AnyType:$out);

  let assemblyFormat = "$in `:` type($in) `to` type($out) attr-dict";
  let hasVerifier = true;
  let hasFolder = true;
}
