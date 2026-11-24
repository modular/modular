// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

mo.graph @broadcast_same_shape() -> (!mo.tensor<[2, 3], f32>) {
  %from   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<2x3xf32>> : !mo.tensor<[2, 3], f32>
  %to     = mo.constant <#M.dense_array<2, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %output = mo.broadcast_to(%from, %to) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  mo.output %output : !mo.tensor<[2, 3], f32>
}
// CHECK{LITERAL}: 'broadcast_same_shape' returned tensor<2x3xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]


mo.graph @broadcast_padding() -> (!mo.tensor<[2, 3], f32>) {
  %from   = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>> : !mo.tensor<[3], f32>
  %to     = mo.constant <#M.dense_array<2, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %output = mo.broadcast_to(%from, %to) : (!mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  mo.output %output : !mo.tensor<[2, 3], f32>
}
// CHECK{LITERAL}: 'broadcast_padding' returned tensor<2x3xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]


mo.graph @broadcast_multiple_axes() -> (!mo.tensor<[2, 2, 2], f32>) {
  %from   = mo.constant <#M.dense_array<1.0, 2.0> : tensor<2x1xf32>> : !mo.tensor<[2, 1], f32>
  %to     = mo.constant <#M.dense_array<2, 2, 2> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %output = mo.broadcast_to(%from, %to) : (!mo.tensor<[2, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  mo.output %output : !mo.tensor<[2, 2, 2], f32>
}
// CHECK{LITERAL}: 'broadcast_multiple_axes' returned tensor<2x2x2xf32> [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]


mo.graph @broadcast_shape_different_ranks() -> (!mo.tensor<[2], si64>) {
  %0      = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1      = mo.constant <#M.dense_array<3> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %output = mo.broadcast_shape(%0, %1) : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  mo.output %output : !mo.tensor<[2], si64>
}
// CHECK{LITERAL}: 'broadcast_shape_different_ranks' returned tensor<2xsi64> [2, 3]


mo.graph @broadcast_shape_commute() -> (!mo.tensor<[2], si64>) {
  %0      = mo.constant <#M.dense_array<3> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %1     = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %output = mo.broadcast_shape(%0, %1) : (!mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2], si64>
  mo.output %output : !mo.tensor<[2], si64>
}
// CHECK{LITERAL}: 'broadcast_shape_commute' returned tensor<2xsi64> [2, 3]


mo.graph @broadcast_shape_4D() -> (!mo.tensor<[4], si32>) {
  %0     = mo.constant <#M.dense_array<2, 1, 1, 1> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %1      = mo.constant <#M.dense_array<1, 3> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %output = mo.broadcast_shape(%0, %1) : (!mo.tensor<[4], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[4], si32>
  mo.output %output : !mo.tensor<[4], si32>
}
// CHECK{LITERAL}: 'broadcast_shape_4D' returned tensor<4xsi32> [2, 1, 1, 3]


mo.graph @broadcast_to_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[1, 2, 1], f32>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  %target_shape = mo.constant <#M.dense_array<10, 20, 2, 30> : tensor<4xsi32>> : !mo.tensor<[4], si32>

  mo.invoke_shape_func["mo.broadcast_to"]<() -> D4, D5, D6, D7>(%input, %target_shape)
    [dataDeptTensors = [1 : i32]] :
    (!mo.tensor<[1, 2, 1], f32>, !mo.tensor<[4], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
// CHECK{LITERAL}: 'broadcast_to_fallback_shape_func' returned tensor<10x20x2x30xsi32>
