// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s


// CHECK-LABEL: mo.graph @static_reshape
//  CHECK-SAME:   <[[N:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>,
mo.graph @static_reshape<N -> out>(
  %input: !mo.tensor<[], si64>,
  %dummy: !mo.tensor<[N], f32>
) -> !mo.tensor<[out], si64> {

  %n = mosh.param.to_value = <N>
  %n_scalar = mo.index.to_tensor(%n) -> !mo.tensor<[], si64>
  %shape = mo.static.reshape(%n_scalar) : !mo.tensor<[], si64> -> !mo.tensor<[1], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[[[N]]], si64>
  %res = mo.broadcast_to<() -> D0>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[D0], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output<D0> %res : !mo.tensor<[D0], si64>
}


// CHECK-LABEL: mo.graph @static_reshape_1d_to_scalar
//  CHECK-SAME:   <[[N:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>,
mo.graph @static_reshape_1d_to_scalar<N -> out>(
  %input: !mo.tensor<[], si64>,
  %dummy: !mo.tensor<[N], f32>
) -> !mo.tensor<[out], si64> {

  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N], f32>) -> !mo.tensor<[1], si64>
  %n = mo.static.reshape(%dummy_shape) : !mo.tensor<[1], si64> -> !mo.tensor<[], si64>
  %start = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
  %step = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  // CHECK: [[RES:%.*]] = mo.range{{.*}} -> !mo.tensor<[[[N]]], si64>
  %res = mo.range<() -> D0>(%start, %n, %step) : (
    !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[D0], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output<D0> %res : !mo.tensor<[D0], si64>
}
