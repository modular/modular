// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

mo.graph @gather_1d_i32_input_2d_indices_axis0() -> (!mo.tensor<[2, 2], si32>) {
  // input:
  //   [1, 2, 3, 4]
  // indices:
  //   [[1, 0],
  //    [2, 1]]
  // expected output:
  //   [[2, 1],
  //    [3, 2]]
  %input   = mo.constant <#M.dense_array<1, 2, 3, 4> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %indices = mo.constant <#M.dense_array<1, 0, 2, 1> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
  %res     = mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[4], si32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2], si32>
  mo.output %res : !mo.tensor<[2, 2], si32>
}
// CHECK{LITERAL}: 'gather_1d_i32_input_2d_indices_axis0' returned tensor<2x2xsi32> [2, 1, 3, 2]

mo.graph @gather_2d_input_2d_indices_axis0() -> (!mo.tensor<[2, 2, 3], f32>) {
  // input:
  //   [[1, 2, 3],
  //    [4, 5, 6],
  //    [7, 8, 9]]
  // indices:
  //   [[1, 0],
  //    [2, 1]]
  // expected output:
  //   [[[4, 5, 6], [1, 2, 3]],
  //    [[7, 8, 9], [4, 5, 6]]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0> : tensor<3x3xf32>> : !mo.tensor<[3, 3], f32>
  %indices = mo.constant <#M.dense_array<1, 0, 2, 1> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
  %res     = mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
  mo.output %res : !mo.tensor<[2, 2, 3], f32>
}
// CHECK{LITERAL}: 'gather_2d_input_2d_indices_axis0' returned tensor<2x2x3xf32> [4.000000e+00, 5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]

mo.graph @gather_2d_input_2d_indices_axis1() -> (!mo.tensor<[3, 2, 2], f32>) {
  // input:
  //   [[1, 2, 3],
  //    [4, 5, 6],
  //    [7, 8, 9]]
  // indices:
  //   [[1, 0],
  //    [2, 1]]
  // expected output:
  //   [[[2, 1], [3, 2]],
  //    [[5, 4], [6, 5]],
  //    [[8, 7], [9, 8]]]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0> : tensor<3x3xf32>> : !mo.tensor<[3, 3], f32>
  %indices = mo.constant <#M.dense_array<1, 0, 2, 1> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
  %res     = mo.gather(%input, %indices) [axis = 1] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[3, 2, 2], f32>
  mo.output %res : !mo.tensor<[3, 2, 2], f32>
}
// CHECK{LITERAL}: 'gather_2d_input_2d_indices_axis1' returned tensor<3x2x2xf32> [2.000000e+00, 1.000000e+00, 3.000000e+00, 2.000000e+00, 5.000000e+00, 4.000000e+00, 6.000000e+00, 5.000000e+00, 8.000000e+00, 7.000000e+00, 9.000000e+00, 8.000000e+00]

mo.graph @gather_2d_input_1d_indices_axis0() -> (!mo.tensor<[4, 3], f32>) {
  // input:
  //   [[1, 2, 3],
  //    [4, 5, 6],
  //    [7, 8, 9]]
  // indices:
  //   [1, 0, 2, 1]
  // expected output:
  //   [[4, 5, 6],
  //    [1, 2, 3],
  //    [7, 8, 9],
  //    [4, 5, 6]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0> : tensor<3x3xf32>> : !mo.tensor<[3, 3], f32>
  %indices = mo.constant <#M.dense_array<1, 0, 2, 1> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %res     = mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[4, 3], f32>
  mo.output %res : !mo.tensor<[4, 3], f32>
}
// CHECK{LITERAL}: 'gather_2d_input_1d_indices_axis0' returned tensor<4x3xf32> [4.000000e+00, 5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]

mo.graph @gather_2d_input_1d_indices_axis1() -> (!mo.tensor<[3, 4], f32>) {
  // input:
  //   [[1, 2, 3],
  //    [4, 5, 6],
  //    [7, 8, 9]]
  // indices:
  //   [1, 0, 2, 1]
  // expected output:
  //   [[2, 1, 3, 2],
  //    [5, 4, 6, 5],
  //    [8, 7, 9, 8]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0> : tensor<3x3xf32>> : !mo.tensor<[3, 3], f32>
  %indices = mo.constant <#M.dense_array<1, 0, 2, 1> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %res     = mo.gather(%input, %indices) [axis = 1] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[3, 4], f32>
  mo.output %res : !mo.tensor<[3, 4], f32>
}
// CHECK{LITERAL}: 'gather_2d_input_1d_indices_axis1' returned tensor<3x4xf32> [2.000000e+00, 1.000000e+00, 3.000000e+00, 2.000000e+00, 5.000000e+00, 4.000000e+00, 6.000000e+00, 5.000000e+00, 8.000000e+00, 7.000000e+00, 9.000000e+00, 8.000000e+00]

mo.graph @gather_2d_input_1d_i64_indices_axis1() -> (!mo.tensor<[3, 4], f32>) {
  // input:
  //   [[1, 2, 3],
  //    [4, 5, 6],
  //    [7, 8, 9]]
  // indices:
  //   [1, 0, 2, 1]
  // expected output:
  //   [[2, 1, 3, 2],
  //    [5, 4, 6, 5],
  //    [8, 7, 9, 8]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0> : tensor<3x3xf32>> : !mo.tensor<[3, 3], f32>
  %indices = mo.constant <#M.dense_array<1, 0, 2, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %res     = mo.gather(%input, %indices) [axis = 1] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[3, 4], f32>
  mo.output %res : !mo.tensor<[3, 4], f32>
}
// CHECK{LITERAL}: 'gather_2d_input_1d_i64_indices_axis1' returned tensor<3x4xf32> [2.000000e+00, 1.000000e+00, 3.000000e+00, 2.000000e+00, 5.000000e+00, 4.000000e+00, 6.000000e+00, 5.000000e+00, 8.000000e+00, 7.000000e+00, 9.000000e+00, 8.000000e+00]


mo.graph @gather_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[1, 2, 3], f32>,
  %indices: !mo.tensor<[4, 5], f32>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  mo.invoke_shape_func["mo.gather"]<() -> D4, D5, D6, D7>(%input, %indices)
    [kgenParams = {axis = -2 : index}, dataDeptTensors = []] :
    (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[4, 5], f32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
// CHECK: 'gather_fallback_shape_func' returned tensor<1x4x5x3xsi32>
