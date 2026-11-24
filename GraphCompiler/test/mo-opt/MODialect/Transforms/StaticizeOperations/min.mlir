// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @min_scalar
//  CHECK-SAME:   <[[N:.*]] -> {{.*}}>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>,
mo.graph @min_scalar<N -> out>(
  %input: !mo.tensor<[], si64>,
  %dummy: !mo.tensor<[N], si64>
) -> !mo.tensor<[out], si64> {

  %n = mosh.param.to_value = <N>
  %n_scalar = mo.index.to_tensor(%n) -> !mo.tensor<[], si64>
  %one = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %min = mo.min(%n_scalar, %one) : !mo.tensor<[], si64>
  %shape = mo.static.reshape(%min) : !mo.tensor<[], si64> -> !mo.tensor<[1], si64>
  
  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[min([[N]], 1)], si64>
  %res = mo.broadcast_to<() -> D0>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[D0], si64>

  // CHECK-NEXT: mo.output{{.*}} [[RES]]
  mo.output<D0> %res : !mo.tensor<[D0], si64>
}


// CHECK-LABEL: mo.graph @min_non_scalar
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]] -> {{.*}}>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>,
mo.graph @min_non_scalar<N1, N2 -> out0, out1>(
  %input: !mo.tensor<[], si64>,
  %dummy: !mo.tensor<[N1, N2], si64>
) -> !mo.tensor<[out0, out1], si64> {

  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, N2], si64>) -> !mo.tensor<[2], si64>
  %const = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %shape = mo.min(%dummy_shape, %const) : !mo.tensor<[2], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[min([[N1]], 1), min([[N2]], 2)], si64>
  %res = mo.broadcast_to<() -> D0, D1>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D0, D1], si64>

  // CHECK-NEXT: mo.output{{.*}} [[RES]]
  mo.output<D0, D1> %res : !mo.tensor<[D0, D1], si64>
}
