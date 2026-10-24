// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

mo.graph @pad_constant_1d() -> (!mo.tensor<[6], f32>) {
  // [1, 2, 3]
  // -->
  // [5, 1, 2, 3, 5, 5]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>> : !mo.tensor<[3], f32>
  %constant = mo.constant <#M.dense_array<5.0> : tensor<f32>> : !mo.tensor<[], f32>
  %paddings = mo.constant <#M.dense_array<1, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %res = mo.pad.constant(%input, %paddings, %constant) : (!mo.tensor<[3], f32>, !mo.tensor<[2], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[6], f32>
  mo.output %res : !mo.tensor<[6], f32>
}
// CHECK{LITERAL}: 'pad_constant_1d' returned tensor<6xf32> [5.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 5.000000e+00, 5.000000e+00]

mo.graph @pad_constant_2d() -> (!mo.tensor<[4, 4], f32>) {
  // [[1, 2],
  //  [3, 4]]
  // --->
  // [[6, 6, 6, 6],
  //  [6, 1, 2, 6],
  //  [6, 3, 4, 6],
  //  [6, 6, 6, 6]]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %constant = mo.constant <#M.dense_array<6.0> : tensor<f32>> : !mo.tensor<[], f32>
  %paddings = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %res = mo.pad.constant(%input, %paddings, %constant) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[4], si64>, !mo.tensor<[], f32>) -> !mo.tensor<[4, 4], f32>
  mo.output %res : !mo.tensor<[4, 4], f32>
}
// CHECK{LITERAL}: 'pad_constant_2d' returned tensor<4x4xf32> [6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00, 6.000000e+00, 6.000000e+00, 3.000000e+00, 4.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00]

mo.graph @pad_constant_2d_dyn_output<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  // [[1, 2],
  //  [3, 4]]
  // --->
  // [[6, 6, 6, 6],
  //  [6, 1, 2, 6],
  //  [6, 3, 4, 6],
  //  [6, 6, 6, 6]]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %constant = mo.constant <#M.dense_array<6.0> : tensor<f32>> : !mo.tensor<[], f32>
  %paddings = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %res = mo.pad.constant<() -> D2, D3>(%input, %paddings, %constant) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[4], si64>, !mo.tensor<[], f32>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], f32>
}
// CHECK{LITERAL}: 'pad_constant_2d_dyn_output' returned tensor<4x4xf32> [6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00, 6.000000e+00, 6.000000e+00, 3.000000e+00, 4.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00]

mo.graph @pad_fallback_shape_func<() -> D0, D1, D2>(
  %input: !mo.tensor<[1, 2, 3], si32>,
  %scalar: !mo.tensor<[], si32>
) -> !mo.tensor<[D0, D1, D2], si32> {

  %paddings = mo.constant <#M.dense_array<1, 0, 2, 3, 5, 4> : tensor<6xsi32>> : !mo.tensor<[6], si32>

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  mo.invoke_shape_func["mo.pad.constant"]<() -> D3, D4, D5>(%input, %paddings, %one)
    [dataDeptTensors = [1 : i32]] :
    (!mo.tensor<[1, 2, 3], si32>, !mo.tensor<[6], si32>, !mo.tensor<[], si32>)

  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D3, D4, D5], si32>

  mo.output<D3, D4, D5> %val : !mo.tensor<[D3, D4, D5], si32>
}
// CHECK: 'pad_fallback_shape_func' returned tensor<2x7x12xsi32>

mo.graph @pad_reflect_1d() -> (!mo.tensor<[6], f32>) {
  // [1, 2, 3]
  // -->
  // [2, 1, 2, 3, 2, 1]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>> : !mo.tensor<[3], f32>
  %paddings = mo.constant <#M.dense_array<1, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %res = mo.pad.reflect(%input, %paddings) : (!mo.tensor<[3], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[6], f32>
  mo.output %res : !mo.tensor<[6], f32>
}
// CHECK{LITERAL}: 'pad_reflect_1d' returned tensor<6xf32> [2.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]

mo.graph @pad_reflect_2d() -> (!mo.tensor<[4, 4], f32>) {
  // [[1, 2],
  //  [3, 4]]
  // --->
  // [[4, 3, 4, 3],
  //  [2, 1, 2, 1],
  //  [4, 3, 4, 3],
  //  [2, 1, 2, 1]]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %paddings = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %res = mo.pad.reflect(%input, %paddings) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[4, 4], f32>
  mo.output %res : !mo.tensor<[4, 4], f32>
}
// CHECK{LITERAL}: 'pad_reflect_2d' returned tensor<4x4xf32> [4.000000e+00, 3.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00, 4.000000e+00, 3.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00]

mo.graph @pad_reflect_2d_dyn_output<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  // [[1, 2],
  //  [3, 4]]
  // --->
  // [[4, 3, 4, 3],
  //  [2, 1, 2, 1],
  //  [4, 3, 4, 3],
  //  [2, 1, 2, 1]]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %paddings = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %res = mo.pad.reflect<() -> D2, D3>(%input, %paddings) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], f32>
}
// CHECK{LITERAL}: 'pad_reflect_2d_dyn_output' returned tensor<4x4xf32> [4.000000e+00, 3.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00, 4.000000e+00, 3.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00]

mo.graph @pad_repeat_1d() -> (!mo.tensor<[6], f32>) {
  // [1, 2, 3]
  // -->
  // [1, 1, 2, 3, 3, 3]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>> : !mo.tensor<[3], f32>
  %paddings = mo.constant <#M.dense_array<1, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %res = mo.pad.repeat(%input, %paddings) : (!mo.tensor<[3], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[6], f32>
  mo.output %res : !mo.tensor<[6], f32>
}
// CHECK{LITERAL}: 'pad_repeat_1d' returned tensor<6xf32> [1.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00]

mo.graph @pad_repeat_2d() -> (!mo.tensor<[4, 4], f32>) {
  // [[1, 2],
  //  [3, 4]]
  // --->
  // [[1, 1, 2, 2],
  //  [1, 1, 2, 2],
  //  [3, 3, 4, 4],
  //  [3, 3, 4, 4]]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %paddings = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %res = mo.pad.repeat(%input, %paddings) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[4, 4], f32>
  mo.output %res : !mo.tensor<[4, 4], f32>
}
// CHECK{LITERAL}: 'pad_repeat_2d' returned tensor<4x4xf32> [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00]

mo.graph @pad_repeat_2d_dyn_output<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  // [[1, 2],
  //  [3, 4]]
  // --->
  // [[1, 1, 2, 2],
  //  [1, 1, 2, 2],
  //  [3, 3, 4, 4],
  //  [3, 3, 4, 4]]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %paddings = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %res = mo.pad.repeat<() -> D2, D3>(%input, %paddings) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], f32>
}
// CHECK{LITERAL}: 'pad_repeat_2d_dyn_output' returned tensor<4x4xf32> [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00]
