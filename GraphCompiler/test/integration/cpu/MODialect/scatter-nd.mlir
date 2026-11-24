// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s


mo.graph @scatter_nd_unsorted_segment_sum() -> (!mo.tensor<[4, 1, 2], f64>) {
  // Implements the special case when lowering tf.UnsortedSegmentSum.
  // input:
  //   [[[1, 2]],
  //    [[3, 4]],
  //    [[5, 6]],
  //    [[7, 8]]]
  // updates:
  //   [[[[1.1, 2.2]]],
  //    [[[3.3, 4.4]]]]
  // indices:
  //   [[[0]],
  //    [[2]]]
  // expected output:
  //   [[[1.1, 2.2]],
  //    [[3, 4]],
  //    [[3.3, 4.4]],
  //    [[7, 8]]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<4x1x2xf64>> : !mo.tensor<[4, 1, 2], f64>
  %updates = mo.constant <#M.dense_array<1.1, 2.2, 3.3, 4.4> : tensor<2x1x1x2xf64>> : !mo.tensor<[2, 1, 1, 2], f64>
  %indices = mo.constant <#M.dense_array<0, 2> : tensor<2x1x1xsi32>> : !mo.tensor<[2, 1, 1], si32>
  %res     = mo.scatter_nd(%input, %updates, %indices) : (!mo.tensor<[4, 1, 2], f64>, !mo.tensor<[2, 1, 1, 2], f64>, !mo.tensor<[2, 1, 1], si32>) -> !mo.tensor<[4, 1, 2], f64>
  mo.output %res : !mo.tensor<[4, 1, 2], f64>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.scatter_nd{{.*}}"
// CHECK{LITERAL}: 'scatter_nd_unsorted_segment_sum' returned tensor<4x1x2xf64> [1.100000e+00, 2.200000e+00, 3.000000e+00, 4.000000e+00, 3.300000e+00, 4.400000e+00, 7.000000e+00, 8.000000e+00]

mo.graph @scatter_nd_fallback_shape_func<() -> D0, D1, D2>() -> !mo.tensor<[D0, D1, D2], si64> {

  %one     = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<4x1x2xf64>> : !mo.tensor<[4, 1, 2], f64>
  %updates = mo.constant <#M.dense_array<1.1, 2.2, 3.3, 4.4> : tensor<2x1x1x2xf64>> : !mo.tensor<[2, 1, 1, 2], f64>
  %indices = mo.constant <#M.dense_array<0, 2> : tensor<2x1x1xsi32>> : !mo.tensor<[2, 1, 1], si32>

  mo.invoke_shape_func["mo.scatter_nd"]<() -> D3, D4, D5>(%input, %updates, %indices)
    [dataDeptTensors = []] : (!mo.tensor<[4, 1, 2], f64>, !mo.tensor<[2, 1, 1, 2], f64>, !mo.tensor<[2, 1, 1], si32>)

  %val = mo.static.broadcast_to(%one) : !mo.tensor<[], si64> -> !mo.tensor<[D3, D4, D5], si64>
  mo.output<D3, D4, D5> %val : !mo.tensor<[D3, D4, D5], si64>
}

// CHECK-LABEL: --- Running 'scatter_nd_fallback_shape_func':
// CHECK-NEXT: 'scatter_nd_fallback_shape_func' returned tensor<4x1x2xsi64>
