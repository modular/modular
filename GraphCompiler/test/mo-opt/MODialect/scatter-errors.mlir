// RUN: mo-opt %s --allow-unregistered-dialect --split-input-file --verify-diagnostics

mo.graph @forward() -> !mo.tensor<[2, 3], si32> {
  %data = mo.constant <#M.dense_array<10, 20, 30, 40, 50, 60> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %updates = mo.constant <#M.dense_array<100, 200, 300, 1000, 2000, 3000> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %indices = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2, 1], si32>

  // expected-error@below {{updates rank must be: data_rank + indices_rank - indices_shape[-1] - 1}}
  %0 = mo.scatter_nd(%data, %updates, %indices) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[6], si32>, !mo.tensor<[2, 1], si32>) -> !mo.tensor<[2, 3], si32>
  mo.output %0 : !mo.tensor<[2, 3], si32>
}

// -----

mo.graph @same_rank(%input : !mo.tensor<[10], si32>, %updates : !mo.tensor<[10, 10], si32>, %indices : !mo.tensor<[10], si64>) -> !mo.tensor<[?], si32> {
  // expected-error@below {{output and inputs (except axis) of a scatter op must have the same rank}}
  %0 = mo.scatter(%input, %updates, %indices) [axis = 0] : (!mo.tensor<[10], si32>, !mo.tensor<[10, 10], si32>, !mo.tensor<[10], si64>) -> !mo.tensor<[?], si32>
  mo.output %0 : !mo.tensor<[?], si32>
}
