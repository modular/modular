// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: @dist_transformer_block_0
// CHECK-LABEL: @caller
// CHECK: rebind{{.*}}input_split_0

// Minimal reproducer for GEX-2691: after PropagateShapes, the mo.call operand
// type may be rewritten to use a different dimension symbol than the callee's
// function type, and the callee's signature is not updated, yielding a
// verification error.
mo.graph @dist_transformer_block_0<input_split_0>(
  %x: !mo.tensor<[input_split_0, 7168], bf16, gpu:0>
) -> !mo.tensor<[input_split_0, 7168], bf16, gpu:0> isSubgraph {
  // Identity body to keep the signature simple and close to the real case.
  mo.output %x : !mo.tensor<[input_split_0, 7168], bf16, gpu:0>
}

// Caller uses a different symbol name for the first dimension.
// The caller starts with a value typed using RMO-style symbols, then uses a
// slice to declare input_split_0 and re-express the tensor with that symbol so
// the call type-checks pre-pass.
// Previously PropagateShapes would bind input_split_0 to the input RMO
// dimension and replace types accordingly, so post-pass the call operand is
// typed with RMO_D0 and the call fails against the callee signature.
mo.graph @caller<RMO_D0>(
  %arg: !mo.tensor<[RMO_D0, 7168], bf16, gpu:0>
) -> !mo.tensor<[input_split_0, 7168], bf16, gpu:0> {
  // Build start/stop/step for a no-op slice that forwards both dims.
  %start = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step  = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_mosh = mosh.param.to_value: !mosh.ape = <[RMO_D0, 7168]>
  %stop = mo.shape.to_tensor(%stop_mosh) -> !mo.tensor<[2], si64>

  // Declare input_split_0 via outputParamDecls while keeping the result type
  // compatible with the callee.
  %v = mo.slice<() -> input_split_0>(%arg, %start, %stop, %step)
     : (!mo.tensor<[RMO_D0, 7168], bf16, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
     -> !mo.tensor<[input_split_0, 7168], bf16, gpu:0>

  // Pre-pass, types match the callee. Post-pass, PropagateShapes previously
  // would input_split_0 with RMO_D0, creating a mismatch here.
  %out = mo.call @dist_transformer_block_0(%v)
       : (!mo.tensor<[input_split_0, 7168], bf16, gpu:0>)
         -> !mo.tensor<[input_split_0, 7168], bf16, gpu:0>
  mo.output %out : !mo.tensor<[input_split_0, 7168], bf16, gpu:0>
}

// CHECK-LABEL: @caller_twice
// CHECK: rebind{{.*}}input_split_0
// Same setup as @caller, but invoke the callee twice to stress that
// normalization doesn’t re-declare the same parameter name in one scope.
mo.graph @caller_twice<RMO_D0>(
  %arg: !mo.tensor<[RMO_D0, 7168], bf16, gpu:0>
) -> !mo.tensor<[input_split_0, 7168], bf16, gpu:0> {
  %start = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step  = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_mosh = mosh.param.to_value: !mosh.ape = <[RMO_D0, 7168]>
  %stop = mo.shape.to_tensor(%stop_mosh) -> !mo.tensor<[2], si64>

  %v = mo.slice<() -> input_split_0>(%arg, %start, %stop, %step)
     : (!mo.tensor<[RMO_D0, 7168], bf16, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
     -> !mo.tensor<[input_split_0, 7168], bf16, gpu:0>

  // Two call sites to ensure we don’t end up redeclaring params.
  %out1 = mo.call @dist_transformer_block_0(%v)
       : (!mo.tensor<[input_split_0, 7168], bf16, gpu:0>)
         -> !mo.tensor<[input_split_0, 7168], bf16, gpu:0>
  %out2 = mo.call @dist_transformer_block_0(%v)
       : (!mo.tensor<[input_split_0, 7168], bf16, gpu:0>)
         -> !mo.tensor<[input_split_0, 7168], bf16, gpu:0>
  mo.output %out2 : !mo.tensor<[input_split_0, 7168], bf16, gpu:0>
}
