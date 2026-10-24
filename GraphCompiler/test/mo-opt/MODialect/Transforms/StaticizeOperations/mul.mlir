// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @mul_scalar
//  CHECK-SAME:   <[[N:.*]] -> {{.*}}>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>,
mo.graph @mul_scalar<N -> out>(
  %input: !mo.tensor<[], si64>,
  %dummy: !mo.tensor<[N], si64>
) -> !mo.tensor<[out], si64> {

  %n = mosh.param.to_value = <N>
  %n_scalar = mo.index.to_tensor(%n) -> !mo.tensor<[], si64>
  %two = mo.constant <#M.dense_array<2> : tensor<si64>> : !mo.tensor<[], si64>
  %mul = mo.mul(%n_scalar, %two) : !mo.tensor<[], si64>
  %shape = mo.static.reshape(%mul) : !mo.tensor<[], si64> -> !mo.tensor<[1], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[mul_no_wrap([[N]], 2)], si64>
  %res = mo.broadcast_to<() -> D0>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[D0], si64>

  // CHECK-NEXT: mo.output{{.*}} [[RES]]
  mo.output<D0> %res : !mo.tensor<[D0], si64>
}


// CHECK-LABEL: mo.graph @mul_non_scalar
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]] -> {{.*}}>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>,
mo.graph @mul_non_scalar<N1, N2 -> out0, out1>(
  %input: !mo.tensor<[], si64>,
  %dummy: !mo.tensor<[N1, N2], si64>
) -> !mo.tensor<[out0, out1], si64> {

  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, N2], si64>) -> !mo.tensor<[2], si64>
  %const = mo.constant <#M.dense_array<2, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %shape = mo.mul(%dummy_shape, %const) : !mo.tensor<[2], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[mul_no_wrap([[N1]], 2), mul_no_wrap([[N2]], 3)], si64>
  %res = mo.broadcast_to<() -> D0, D1>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D0, D1], si64>

  // CHECK-NEXT: mo.output{{.*}} [[RES]]
  mo.output<D0, D1> %res : !mo.tensor<[D0, D1], si64>
}
