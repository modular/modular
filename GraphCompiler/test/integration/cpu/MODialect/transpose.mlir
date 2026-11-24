// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// TODO(GEX-1573): shape function.

mo.graph @transpose_2d_identity() -> (!mo.tensor<[2, 3], f32>) {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<2x3xf32>> : !mo.tensor<[2, 3], f32>
  %perm  = mo.constant <#M.dense_array<0, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res   = mo.transpose(%input, %perm) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}
// CHECK{LITERAL}: 'transpose_2d_identity' returned tensor<2x3xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]

mo.graph @transpose_2d() -> (!mo.tensor<[3, 2], f32>) {
  // [[1, 2, 3]
  //  [4, 5, 6]]
  // --->
  // [[1, 4],
  //  [2, 5],
  //  [3, 6]]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<2x3xf32>> : !mo.tensor<[2, 3], f32>
  %perm  = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res   = mo.transpose(%input, %perm) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}
// CHECK{LITERAL}: 'transpose_2d' returned tensor<3x2xf32> [1.000000e+00, 4.000000e+00, 2.000000e+00, 5.000000e+00, 3.000000e+00, 6.000000e+00]

mo.graph @transpose_3d_identity() -> (!mo.tensor<[2, 2, 3], f32>) {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %perm  = mo.constant <#M.dense_array<0, 1, 2> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %res   = mo.transpose(%input, %perm) : (!mo.tensor<[2, 2, 3], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 3], f32>
  mo.output %res : !mo.tensor<[2, 2, 3], f32>
}
// CHECK{LITERAL}: 'transpose_3d_identity' returned tensor<2x2x3xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]

mo.graph @transpose_3d() -> (!mo.tensor<[3, 2, 2], f32>) {
  // [[[1, 2, 3]
  //   [4, 5, 6]],
  //  [[7, 8, 9]
  //   [10, 11, 12]]]
  // --->
  // [[[1, 4],
  //   [7, 10]],
  //  [[2, 5],
  //   [8, 11]]
  //  [[3, 6],
  //   [9, 12]]]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %perm  = mo.constant <#M.dense_array<2, 0, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %res   = mo.transpose(%input, %perm) : (!mo.tensor<[2, 2, 3], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[3, 2, 2], f32>
  mo.output %res : !mo.tensor<[3, 2, 2], f32>
}
// CHECK{LITERAL}: 'transpose_3d' returned tensor<3x2x2xf32> [1.000000e+00, 4.000000e+00, 7.000000e+00, 1.000000e+01, 2.000000e+00, 5.000000e+00, 8.000000e+00, 1.100000e+01, 3.000000e+00, 6.000000e+00, 9.000000e+00, 1.200000e+01]

mo.graph @transpose_2d_dyn_output<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  // [[1, 2, 3]
  //  [4, 5, 6]]
  // --->
  // [[1, 4],
  //  [2, 5],
  //  [3, 6]]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<2x3xf32>> : !mo.tensor<[2, 3], f32>
  %perm  = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res   = mo.transpose<() -> D2, D3>(%input, %perm) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], f32>
}
// CHECK{LITERAL}: 'transpose_2d_dyn_output' returned tensor<3x2xf32> [1.000000e+00, 4.000000e+00, 2.000000e+00, 5.000000e+00, 3.000000e+00, 6.000000e+00]


mo.graph @transpose_shape(%arg0: !mo.tensor<[1, 2, 3, 4], f32>) -> !mo.tensor<[D1, D2, D3, D4], f32> {
  %perm = mo.constant <#M.dense_array<0, 3, 2, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  mo.invoke_shape_func["mo.transpose"]<() -> D1, D2, D3, D4>(%arg0, %perm) [dataDeptTensors = [1 : i32]] : (!mo.tensor<[1, 2, 3, 4], f32>, !mo.tensor<[4], si64>)

  %dummy = mo.constant <#M.dense_array<0.1> : tensor<f32>> : !mo.tensor<[], f32>
  %bcast = mo.static.broadcast_to(%dummy): !mo.tensor<[], f32> -> !mo.tensor<[D1, D2, D3, D4], f32>
  mo.output %bcast : !mo.tensor<[D1, D2, D3, D4], f32>
}
// CHECK-LABEL:--- Running 'transpose_shape':
// CHECK: 'transpose_shape' returned tensor<1x4x3x2xf32>


mo.graph @transpose_shape_2(%arg0: !mo.tensor<[1, 2, 3, 4], f32>) -> !mo.tensor<[D1, D2, D3, D4], f32> {
  %perm = mo.constant <#M.dense_array<3, 2, 1, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  mo.invoke_shape_func["mo.transpose"]<() -> D1, D2, D3, D4>(%arg0, %perm) [dataDeptTensors = [1 : i32]] : (!mo.tensor<[1, 2, 3, 4], f32>, !mo.tensor<[4], si64>)

  %dummy = mo.constant <#M.dense_array<0.1> : tensor<f32>> : !mo.tensor<[], f32>
  %bcast = mo.static.broadcast_to(%dummy): !mo.tensor<[], f32> -> !mo.tensor<[D1, D2, D3, D4], f32>
  mo.output %bcast : !mo.tensor<[D1, D2, D3, D4], f32>
}
// CHECK-LABEL:--- Running 'transpose_shape_2':
// CHECK: 'transpose_shape_2' returned tensor<4x3x2x1xf32>
