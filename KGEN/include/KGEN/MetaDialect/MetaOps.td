//===- MetaOps.td ---------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the Meta dialect.
//
//===----------------------------------------------------------------------===//

include "KGEN/KGENDialect/KGENDialect.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// BufferSizeOp
//===----------------------------------------------------------------------===//

def BufferSizeOp : MetaOp<"buffer.size", [NoSideEffect]> {
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

//===----------------------------------------------------------------------===//
// BufferDTypeOp
//===----------------------------------------------------------------------===//

def BufferDTypeOp : MetaOp<"buffer.dtype", [NoSideEffect]> {
  let summary = [{
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
  let summary = [{
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
// BufferCastOp
//===----------------------------------------------------------------------===//

def BufferCastOp : MetaOp<"buffer.cast", [NoSideEffect]> {
  let summary = [{
    Cast the input buffer to a buffer of the specified type.

    Example:
    ```mlir
    %0 = meta.buffer.cast %unknown: !buffer<?x?> to !buffer<?xf32>
    %1 = meta.buffer.cast %other: !buffer<1023xf32> to !buffer<?xf32>
    ```
  }];

  let arguments = (ins BufferType:$buffer);
  let results = (outs BufferType:$result);
  let assemblyFormat = [{
    $buffer `:` qualified(type($buffer)) `to`
     qualified(type($result)) attr-dict
  }];
  let hasVerifier = true;
  let hasFolder = true;
}

//===----------------------------------------------------------------------===//
// MetaCastToBuiltinOp
//===----------------------------------------------------------------------===//

def MetaCastToBuiltinOp : MetaOp<"cast_to_builtin", [NoSideEffect]> {
  let summary = [{
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
  let summary = [{
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
