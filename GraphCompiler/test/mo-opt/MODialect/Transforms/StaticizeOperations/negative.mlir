// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @negative_scalar
//  CHECK-SAME:   <[[N:.*]] -> {{.*}}>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], f32>,
mo.graph @negative_scalar<N -> out>(
  %input: !mo.tensor<[], f32>,
  %dummy: !mo.tensor<[N], f32>
) -> !mo.tensor<[out], f32> {

  %n = mosh.param.to_value = <N>
  %n_scalar = mo.index.to_tensor(%n) -> !mo.tensor<[], si64>
  %ten = mo.constant <#M.dense_array<10> : tensor<si64>> : !mo.tensor<[], si64>
  %negative = mo.negative(%n_scalar) : !mo.tensor<[], si64>
  %sum = mo.add(%negative, %ten) : !mo.tensor<[], si64>
  %shape = mo.static.reshape(%sum) : !mo.tensor<[], si64> -> !mo.tensor<[1], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[add(mul_no_wrap([[N]], -1), 10)], f32>
  %res = mo.broadcast_to<() -> D0>(%input, %shape) : (
    !mo.tensor<[], f32>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[D0], f32>

  // CHECK-NEXT: mo.output{{.*}} [[RES]]
  mo.output<D0> %res : !mo.tensor<[D0], f32>
}


// CHECK-LABEL: mo.graph @negative_non_scalar
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]] -> {{.*}}>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], f32>,
mo.graph @negative_non_scalar<N1, N2 -> O1, O2>(
  %input: !mo.tensor<[], f32>,
  %dummy: !mo.tensor<[N1, N2], f32>
) -> !mo.tensor<[O1, O2], f32> {

  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, N2], f32>) -> !mo.tensor<[2], si64>
  %const = mo.constant <#M.dense_array<20, 30> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %negative = mo.negative(%dummy_shape) : !mo.tensor<[2], si64>
  %shape = mo.add(%negative, %const) : !mo.tensor<[2], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[add(mul_no_wrap([[N1]], -1), 20), add(mul_no_wrap([[N2]], -1), 30)], f32>
  %res = mo.broadcast_to<() -> D0, D1>(%input, %shape) : (
    !mo.tensor<[], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D0, D1], f32>

  // CHECK-NEXT: mo.output{{.*}} [[RES]]
  mo.output<D0, D1> %res : !mo.tensor<[D0, D1], f32>
}
