// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @gather_scalar_indices
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>,
mo.graph @gather_scalar_indices<N1, N2 -> out>(
  %input: !mo.tensor<[], si64>,
  %dummy: !mo.tensor<[N1, 10, N2], f32>
) -> !mo.tensor<[out], si64> {

  %idx = mo.constant <#M.dense_array<-1> : tensor<si64>> : !mo.tensor<[], si64>
  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, 10, N2], f32>) -> !mo.tensor<[3], si64>
  %n2 = mo.gather(%dummy_shape, %idx) [axis = 0] : (
    !mo.tensor<[3], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[], si64>
  %shape = mo.static.reshape(%n2) : !mo.tensor<[], si64> -> !mo.tensor<[1], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[[[N2]]], si64>
  %res = mo.broadcast_to<() -> D0>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[D0], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output<D0> %res : !mo.tensor<[D0], si64>
}


// CHECK-LABEL: mo.graph @gather_1d_indices
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>,
mo.graph @gather_1d_indices<N1, N2 -> out0, out1>(
  %input: !mo.tensor<[], si64>,
  %dummy: !mo.tensor<[N1, 10, N2], f32>
) -> !mo.tensor<[out0, out1], si64> {

  %indices = mo.constant <#M.dense_array<-1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, 10, N2], f32>) -> !mo.tensor<[3], si64>
  %shape = mo.gather(%dummy_shape, %indices) [axis = 0] : (
    !mo.tensor<[3], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[[[N2]], 10], si64>
  %res = mo.broadcast_to<() -> D0, D1>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D0, D1], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output<D0, D1> %res : !mo.tensor<[D0, D1], si64>
}
