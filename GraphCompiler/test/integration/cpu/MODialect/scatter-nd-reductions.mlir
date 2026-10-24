// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

mo.graph @scatter_nd_add() -> (!mo.tensor<[4, 1, 2], f64>) {
  // Implements the special case when lowering tf.UnsortedSegmentSum.
  // input:
  //   [[[1, 2]],
  //    [[3, 4]],
  //    [[5, 6]],
  //    [[7, 8]]]
  // updates:
  //   [[[[1.1, 2.2]]],
  //    [[[3.3, 4.4]]],
  //    [[[5.5, 6.6]]]]
  // indices:
  //   [[[0]],
  //    [[0]],
  //    [[2]]]
  // expected output:
  //   [[[5.4, 8.6]],
  //    [[3,   4]],
  //    [[10.5, 12.6]],
  //    [[7,   8]]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<4x1x2xf64>> : !mo.tensor<[4, 1, 2], f64>
  %updates = mo.constant <#M.dense_array<1.1, 2.2, 3.3, 4.4, 5.5, 6.6> : tensor<3x1x1x2xf64>> : !mo.tensor<[3, 1, 1, 2], f64>
  %indices = mo.constant <#M.dense_array<0, 0, 2> : tensor<3x1x1xsi32>> : !mo.tensor<[3, 1, 1], si32>
  %res     = mo.scatter_nd.add(%input, %updates, %indices) : (!mo.tensor<[4, 1, 2], f64>, !mo.tensor<[3, 1, 1, 2], f64>, !mo.tensor<[3, 1, 1], si32>) -> !mo.tensor<[4, 1, 2], f64>
  mo.output %res : !mo.tensor<[4, 1, 2], f64>
}
// CHECK{LITERAL}: 'scatter_nd_add' returned tensor<4x1x2xf64> [5.400000e+00, 8.600000e+00, 3.000000e+00, 4.000000e+00, 1.050000e+01, 1.260000e+01, 7.000000e+00, 8.000000e+00]

mo.graph @scatter_nd_max() -> (!mo.tensor<[4, 1, 2], f64>) {
  // Implements the special case when lowering tf.UnsortedSegmentSum.
  // input:
  //   [[[1, 2]],
  //    [[3, 4]],
  //    [[5, 6]],
  //    [[7, 8]]]
  // updates:
  //   [[[[1.1, 2.2]]],
  //    [[[3.3, 4.4]]],
  //    [[[5.5, 6.6]]]]
  // indices:
  //   [[[0]],
  //    [[0]],
  //    [[2]]]
  // expected output:
  //   [[[3.3, 4.4]],
  //    [[3,   4]],
  //    [[5.5, 6.6]],
  //    [[7,   8]]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<4x1x2xf64>> : !mo.tensor<[4, 1, 2], f64>
  %updates = mo.constant <#M.dense_array<1.1, 2.2, 3.3, 4.4, 5.5, 6.6> : tensor<3x1x1x2xf64>> : !mo.tensor<[3, 1, 1, 2], f64>
  %indices = mo.constant <#M.dense_array<0, 0, 2> : tensor<3x1x1xsi32>> : !mo.tensor<[3, 1, 1], si32>
  %res     = mo.scatter_nd.max(%input, %updates, %indices) : (!mo.tensor<[4, 1, 2], f64>, !mo.tensor<[3, 1, 1, 2], f64>, !mo.tensor<[3, 1, 1], si32>) -> !mo.tensor<[4, 1, 2], f64>
  mo.output %res : !mo.tensor<[4, 1, 2], f64>
}
// CHECK{LITERAL}: 'scatter_nd_max' returned tensor<4x1x2xf64> [3.300000e+00, 4.400000e+00, 3.000000e+00, 4.000000e+00, 5.500000e+00, 6.600000e+00, 7.000000e+00, 8.000000e+00]

mo.graph @scatter_nd_min() -> (!mo.tensor<[4, 1, 2], f64>) {
  // Implements the special case when lowering tf.UnsortedSegmentSum.
  // input:
  //   [[[1, 2]],
  //    [[3, 4]],
  //    [[5, 6]],
  //    [[7, 8]]]
  // updates:
  //   [[[[1.1, 2.2]]],
  //    [[[3.3, 4.4]]],
  //    [[[5.5, 6.6]]]]
  // indices:
  //   [[[0]],
  //    [[0]],
  //    [[2]]]
  // expected output:
  //   [[[1, 2]],
  //    [[3, 4]],
  //    [[5, 6]],
  //    [[7, 8]]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<4x1x2xf64>> : !mo.tensor<[4, 1, 2], f64>
  %updates = mo.constant <#M.dense_array<1.1, 2.2, 3.3, 4.4, 5.5, 6.6> : tensor<3x1x1x2xf64>> : !mo.tensor<[3, 1, 1, 2], f64>
  %indices = mo.constant <#M.dense_array<0, 0, 2> : tensor<3x1x1xsi32>> : !mo.tensor<[3, 1, 1], si32>
  %res     = mo.scatter_nd.min(%input, %updates, %indices) : (!mo.tensor<[4, 1, 2], f64>, !mo.tensor<[3, 1, 1, 2], f64>, !mo.tensor<[3, 1, 1], si32>) -> !mo.tensor<[4, 1, 2], f64>
  mo.output %res : !mo.tensor<[4, 1, 2], f64>
}
// CHECK{LITERAL}: 'scatter_nd_min' returned tensor<4x1x2xf64> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]

mo.graph @scatter_nd_mul() -> (!mo.tensor<[4, 1, 2], f64>) {
  // Implements the special case when lowering tf.UnsortedSegmentSum.
  // input:
  //   [[[1, 2]],
  //    [[3, 4]],
  //    [[5, 6]],
  //    [[7, 8]]]
  // updates:
  //   [[[[1.1, 2.2]]],
  //    [[[3.3, 4.4]]],
  //    [[[5.5, 6.6]]]]
  // indices:
  //   [[[0]],
  //    [[0]],
  //    [[2]]]
  // expected output:
  //   [[[3.63, 19.36]],
  //    [[3,    4]],
  //    [[27.5, 39.6]],
  //    [[7,    8]]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<4x1x2xf64>> : !mo.tensor<[4, 1, 2], f64>
  %updates = mo.constant <#M.dense_array<1.1, 2.2, 3.3, 4.4, 5.5, 6.6> : tensor<3x1x1x2xf64>> : !mo.tensor<[3, 1, 1, 2], f64>
  %indices = mo.constant <#M.dense_array<0, 0, 2> : tensor<3x1x1xsi32>> : !mo.tensor<[3, 1, 1], si32>
  %res     = mo.scatter_nd.mul(%input, %updates, %indices) : (!mo.tensor<[4, 1, 2], f64>, !mo.tensor<[3, 1, 1, 2], f64>, !mo.tensor<[3, 1, 1], si32>) -> !mo.tensor<[4, 1, 2], f64>
  mo.output %res : !mo.tensor<[4, 1, 2], f64>
}
// CHECK{LITERAL}: 'scatter_nd_mul' returned tensor<4x1x2xf64> [3.630000e+00, 1.936000e+01, 3.000000e+00, 4.000000e+00, 2.750000e+01, 3.960000e+01, 7.000000e+00, 8.000000e+00]
