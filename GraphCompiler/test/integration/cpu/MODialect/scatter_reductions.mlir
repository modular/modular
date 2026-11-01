// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

mo.graph @scatter_add() -> (!mo.tensor<[4, 2], f64>) {
  // Implements the special case when lowering tf.UnsortedSegmentSum.
  // input:
  //   [[1, 2],
  //    [3, 4],
  //    [5, 6],
  //    [7, 8]]
  // updates:
  //   [[1.1, 2.2]
  //    [3.3, 4.4]]
  // indices:
  //   [[0, 1]
  //    [3, 2]]
  // axis:
  //   [0]
  // expected output:
  //   [[2.1,  2],
  //    [3,    6.2],
  //    [5,    10.4],
  //    [10.3, 8]]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<4x2xf64>> : !mo.tensor<[4, 2], f64>
  %updates = mo.constant <#M.dense_array<1.1, 2.2, 3.3, 4.4> : tensor<2x2xf64>> : !mo.tensor<[2, 2], f64>
  %indices = mo.constant <#M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
  %axis    = mo.constant <#M.dense_array<0> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %res     = mo.scatter.add(%input, %updates, %indices, %axis) : (
    !mo.tensor<[4, 2], f64>, !mo.tensor<[2, 2], f64>, !mo.tensor<[2, 2], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[4, 2], f64>
  mo.output %res : !mo.tensor<[4, 2], f64>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.scatter{{.*}}"
// CHECK{LITERAL}: 'scatter_add' returned tensor<4x2xf64> [2.100000e+00, 2.000000e+00, 3.000000e+00, 6.200000e+00, 5.000000e+00, 1.040000e+01, 1.030000e+01, 8.000000e+00]

mo.graph @scatter_max() -> (!mo.tensor<[4, 2], f64>) {
  // Implements the special case when lowering tf.UnsortedSegmentSum.
  // input:
  //   [[1, 2],
  //    [3, 4],
  //    [5, 6],
  //    [7, 8]]
  // updates:
  //   [[1.1, 2.2]
  //    [3.3, 4.4]]
  // indices:
  //   [[0, 1]
  //    [3, 2]]
  // axis:
  //   [0]
  // expected output:
  //   [[1.1, 2],
  //    [3,   4],
  //    [5,   6],
  //    [7,   8]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<4x2xf64>> : !mo.tensor<[4, 2], f64>
  %updates = mo.constant <#M.dense_array<1.1, 2.2, 3.3, 4.4> : tensor<2x2xf64>> : !mo.tensor<[2, 2], f64>
  %indices = mo.constant <#M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
  %axis    = mo.constant <#M.dense_array<0> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %res     = mo.scatter.max(%input, %updates, %indices, %axis) : (
    !mo.tensor<[4, 2], f64>, !mo.tensor<[2, 2], f64>, !mo.tensor<[2, 2], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[4, 2], f64>
  mo.output %res : !mo.tensor<[4, 2], f64>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.scatter.max{{.*}}"
// CHECK{LITERAL}: 'scatter_max' returned tensor<4x2xf64> [1.100000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]

mo.graph @scatter_min() -> (!mo.tensor<[4, 2], f64>) {
  // Implements the special case when lowering tf.UnsortedSegmentSum.
  // input:
  //   [[1, 2],
  //    [3, 4],
  //    [5, 6],
  //    [7, 8]]
  // updates:
  //   [[1.1, 2.2]
  //    [3.3, 4.4]]
  // indices:
  //   [[0, 1]
  //    [3, 2]]
  // axis:
  //   [0]
  // expected output:
  //   [[1,   2],
  //    [3,   2.2],
  //    [5,   4.4],
  //    [3.3, 8]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<4x2xf64>> : !mo.tensor<[4, 2], f64>
  %updates = mo.constant <#M.dense_array<1.1, 2.2, 3.3, 4.4> : tensor<2x2xf64>> : !mo.tensor<[2, 2], f64>
  %indices = mo.constant <#M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
  %axis    = mo.constant <#M.dense_array<0> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %res     = mo.scatter.min(%input, %updates, %indices, %axis) : (
    !mo.tensor<[4, 2], f64>, !mo.tensor<[2, 2], f64>, !mo.tensor<[2, 2], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[4, 2], f64>
  mo.output %res : !mo.tensor<[4, 2], f64>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.scatter.min{{.*}}"
// CHECK{LITERAL}: 'scatter_min' returned tensor<4x2xf64> [1.000000e+00, 2.000000e+00, 3.000000e+00, 2.200000e+00, 5.000000e+00, 4.400000e+00, 3.300000e+00, 8.000000e+00]

mo.graph @scatter_mul() -> (!mo.tensor<[4, 2], f64>) {
  // Implements the special case when lowering tf.UnsortedSegmentSum.
  // input:
  //   [[1, 2],
  //    [3, 4],
  //    [5, 6],
  //    [7, 8]]
  // updates:
  //   [[1.1, 2.2]
  //    [3.3, 4.4]]
  // indices:
  //   [[0, 1]
  //    [3, 2]]
  // axis:
  //   [0]
  // expected output:
  //   [[1.1,  2],
  //    [3,    8.8],
  //    [5,    26.4],
  //    [23.1, 8]]]
  %input   = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<4x2xf64>> : !mo.tensor<[4, 2], f64>
  %updates = mo.constant <#M.dense_array<1.1, 2.2, 3.3, 4.4> : tensor<2x2xf64>> : !mo.tensor<[2, 2], f64>
  %indices = mo.constant <#M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
  %axis    = mo.constant <#M.dense_array<0> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %res     = mo.scatter.mul(%input, %updates, %indices, %axis) : (
    !mo.tensor<[4, 2], f64>, !mo.tensor<[2, 2], f64>, !mo.tensor<[2, 2], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[4, 2], f64>
  mo.output %res : !mo.tensor<[4, 2], f64>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.scatter{{.*}}"
// CHECK{LITERAL}: 'scatter_mul' returned tensor<4x2xf64> [1.100000e+00, 2.000000e+00, 3.000000e+00, 8.800000e+00, 5.000000e+00, 2.640000e+01, 2.310000e+01, 8.000000e+00]

// These should all share the same fallback shape func
mo.graph @scatter_fallback_shape_func<() -> D0, D1, D2>(
  %input: !mo.tensor<[1, 2, 3], f32>,
  %updates: !mo.tensor<[1, 1, 2], f32>,
  %indices: !mo.tensor<[1, 1, 2], si32>
) -> !mo.tensor<[D0, D1, D2], si32> {

  %axis = mo.constant <#M.dense_array<-1> : tensor<si32>> : !mo.tensor<[], si32>

  mo.invoke_shape_func["mo.scatter.min"]<() -> D3, D4, D5>(%input, %updates, %indices, %axis)
    [dataDeptTensors = [3 : i32]] :
    (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 1, 2], f32>, !mo.tensor<[1, 1, 2], si32>, !mo.tensor<[], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D3, D4, D5], si32>

  mo.output<D3, D4, D5> %val : !mo.tensor<[D3, D4, D5], si32>
}
// CHECK-LABEL: --- Running 'scatter_fallback_shape_func':
// CHECK-NEXT: 'scatter_fallback_shape_func' returned tensor<1x2x3xsi32>
