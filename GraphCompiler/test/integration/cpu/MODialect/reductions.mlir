// RUN: mo-opt %s --mo-shape-inference --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// Note: To hit as many options in as few tests as possible, always doing negative axis

// input:
//   [[[1, 2, 3],
//     [4, 5, 6]],
//    [[-1, -2, -3],
//     [-4, -5, -6]]]
//

mo.graph @mean_rank3_first_axis() -> !mo.tensor<[1, 2, 3], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %sum   = mo.reduce.add(%input) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32>
  %len   = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %len_b = mo.static.broadcast_to(%len) : !mo.tensor<[], f32> -> !mo.tensor<[1, 2, 3], f32>
  %res   = mo.div(%sum, %len_b) : !mo.tensor<[1, 2, 3], f32>
  mo.output %res : !mo.tensor<[1, 2, 3], f32>
}
// CHECK{LITERAL}: 'mean_rank3_first_axis' returned tensor<1x2x3xf32> [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]

mo.graph @mean_rank3_second_axis() -> !mo.tensor<[2, 1, 3], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %sum   = mo.reduce.add(%input) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  %len   = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %len_b = mo.static.broadcast_to(%len) : !mo.tensor<[], f32> -> !mo.tensor<[2, 1, 3], f32>
  %res   = mo.div(%sum, %len_b) : !mo.tensor<[2, 1, 3], f32>
  mo.output %res : !mo.tensor<[2, 1, 3], f32>
}
// CHECK{LITERAL}: 'mean_rank3_second_axis' returned tensor<2x1x3xf32> [2.500000e+00, 3.500000e+00, 4.500000e+00, -2.500000e+00, -3.500000e+00, -4.500000e+00]

mo.graph @mean_rank3_last_axis() -> !mo.tensor<[2, 2, 1], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %sum   = mo.reduce.add(%input) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 2, 1], f32>
  %len   = mo.constant <#M.dense_array<3.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %len_b = mo.static.broadcast_to(%len) : !mo.tensor<[], f32> -> !mo.tensor<[2, 2, 1], f32>
  %res   = mo.div(%sum, %len_b) : !mo.tensor<[2, 2, 1], f32>
  mo.output %res : !mo.tensor<[2, 2, 1], f32>
}
// CHECK{LITERAL}: 'mean_rank3_last_axis' returned tensor<2x2x1xf32> [2.000000e+00, 5.000000e+00, -2.000000e+00, -5.000000e+00]

mo.graph @sum_rank3_first_axis() -> !mo.tensor<[1, 2, 3], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %res   = mo.reduce.add(%input) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32>
  mo.output %res : !mo.tensor<[1, 2, 3], f32>
}
// CHECK{LITERAL}: 'sum_rank3_first_axis' returned tensor<1x2x3xf32> [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]

mo.graph @sum_rank3_second_axis() -> !mo.tensor<[2, 1, 3], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %res   = mo.reduce.add(%input) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %res : !mo.tensor<[2, 1, 3], f32>
}
// CHECK{LITERAL}: 'sum_rank3_second_axis' returned tensor<2x1x3xf32> [5.000000e+00, 7.000000e+00, 9.000000e+00, -5.000000e+00, -7.000000e+00, -9.000000e+00]

mo.graph @sum_rank3_last_axis() -> !mo.tensor<[2, 2, 1], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %res   = mo.reduce.add(%input) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 2, 1], f32>
  mo.output %res : !mo.tensor<[2, 2, 1], f32>
}
// CHECK{LITERAL}: 'sum_rank3_last_axis' returned tensor<2x2x1xf32> [6.000000e+00, 1.500000e+01, -6.000000e+00, -1.500000e+01]

mo.graph @prod_rank3_first_axis<() -> D0, D1, D2>() -> !mo.tensor<[D0, D1, D2], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %res   = mo.reduce.mul<() -> D3, D4, D5>(%input) [axis = -3] : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[D3, D4, D5], f32>
  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK{LITERAL}: 'prod_rank3_first_axis' returned tensor<1x2x3xf32> [-1.000000e+00, -4.000000e+00, -9.000000e+00, -1.600000e+01, -2.500000e+01, -3.600000e+01]

mo.graph @prod_rank3_second_axis<() -> D0, D1, D2>() -> !mo.tensor<[D0, D1, D2], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %res   = mo.reduce.mul<() -> D3, D4, D5>(%input) [axis = -2] : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[D3, D4, D5], f32>
  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK{LITERAL}: 'prod_rank3_second_axis' returned tensor<2x1x3xf32> [4.000000e+00, 1.000000e+01, 1.800000e+01, 4.000000e+00, 1.000000e+01, 1.800000e+01]

mo.graph @prod_rank3_last_axis<() -> D0, D1, D2>() -> !mo.tensor<[D0, D1, D2], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %res   = mo.reduce.mul<() -> D3, D4, D5>(%input) [axis = -1] : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[D3, D4, D5], f32>
  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK{LITERAL}: 'prod_rank3_last_axis' returned tensor<2x2x1xf32> [6.000000e+00, 1.200000e+02, -6.000000e+00, -1.200000e+02]

mo.graph @min_rank3_first_axis() -> !mo.tensor<[1, 2, 3], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %res   = mo.reduce.min(%input) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32>
  mo.output %res : !mo.tensor<[1, 2, 3], f32>
}
// CHECK{LITERAL}: 'min_rank3_first_axis' returned tensor<1x2x3xf32> [-1.000000e+00, -2.000000e+00, -3.000000e+00, -4.000000e+00, -5.000000e+00, -6.000000e+00]

mo.graph @min_rank3_second_axis() -> !mo.tensor<[2, 1, 3], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %res   = mo.reduce.min(%input) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %res : !mo.tensor<[2, 1, 3], f32>
}
// CHECK{LITERAL}: 'min_rank3_second_axis' returned tensor<2x1x3xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, -4.000000e+00, -5.000000e+00, -6.000000e+00]

mo.graph @min_rank3_last_axis() -> !mo.tensor<[2, 2, 1], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %res   = mo.reduce.min(%input) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 2, 1], f32>
  mo.output %res : !mo.tensor<[2, 2, 1], f32>
}
// CHECK{LITERAL}: 'min_rank3_last_axis' returned tensor<2x2x1xf32> [1.000000e+00, 4.000000e+00, -3.000000e+00, -6.000000e+00]

mo.graph @max_rank3_first_axis() -> !mo.tensor<[1, 2, 3], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %res   = mo.reduce.max(%input) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32>
  mo.output %res : !mo.tensor<[1, 2, 3], f32>
}
// CHECK{LITERAL}: 'max_rank3_first_axis' returned tensor<1x2x3xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]

mo.graph @max_rank3_second_axis() -> !mo.tensor<[2, 1, 3], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %res   = mo.reduce.max(%input) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %res : !mo.tensor<[2, 1, 3], f32>
}
// CHECK{LITERAL}: 'max_rank3_second_axis' returned tensor<2x1x3xf32> [4.000000e+00, 5.000000e+00, 6.000000e+00, -1.000000e+00, -2.000000e+00, -3.000000e+00]

mo.graph @max_rank3_last_axis() -> !mo.tensor<[2, 2, 1], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %res   = mo.reduce.max(%input) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 2, 1], f32>
  mo.output %res : !mo.tensor<[2, 2, 1], f32>
}
// CHECK{LITERAL}: 'max_rank3_last_axis' returned tensor<2x2x1xf32> [3.000000e+00, 6.000000e+00, -1.000000e+00, -4.000000e+00]

mo.graph @arg_max_rank2_last_axis<() -> D0, D1>() -> !mo.tensor<[D0, D1], si64> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0> : tensor<4x3xf32>> : !mo.tensor<[4, 3], f32>
  %res   = mo.reduce.arg_max<() -> D2, D3>(%input) [axis = -1] : (!mo.tensor<[4, 3], f32>) -> !mo.tensor<[D2, D3], si64>
  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], si64>
}
// CHECK{LITERAL}: 'arg_max_rank2_last_axis' returned tensor<4x1xsi64> [2, 2, 0, 0]
