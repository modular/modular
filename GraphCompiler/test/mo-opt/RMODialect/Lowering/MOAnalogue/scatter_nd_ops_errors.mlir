// RUN: not mo-opt --mo-legalize-rmo-operators %s 2>&1 | FileCheck %s

mo.graph @forward() -> !mo.tensor<[2, 3], si32> {
  %data = mo.constant <#M.dense_array<10, 20, 30, 40, 50, 60> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %updates = mo.constant <#M.dense_array<100, 200, 300, 1000, 2000, 3000> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %indices = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2, 1], si32>

  // NOTE: we do not use --verify-diagnostics as other diagnostics are emitted
  // and we seem to need to check all of them.
  //
  // The rank mismatch surfaces directly from the MO scatter_nd verifier on
  // the lowered op (which receives the user-typed `[6]` updates operand
  // verbatim).
  // CHECK: error: updates rank must be: data_rank + indices_rank - indices_shape[-1] - 1
  %0 = rmo.mo.scatter_nd(%data, %updates, %indices) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[6], si32>, !mo.tensor<[2, 1], si32>) -> !mo.tensor<[2, 3], si32>
  mo.output %0 : !mo.tensor<[2, 3], si32>
}
