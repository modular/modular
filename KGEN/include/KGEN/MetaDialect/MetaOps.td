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
