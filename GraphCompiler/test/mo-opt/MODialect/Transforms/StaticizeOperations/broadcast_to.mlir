// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @broadcast_to_from_scalar
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>
mo.graph @broadcast_to_from_scalar<() -> O0, O1, O2, O3>(
  %input: !mo.tensor<[], si64>
) -> !mo.tensor<[O0, O1, O2, O3], si64> {

  %one = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %shape_shape = mo.constant <#M.dense_array<4> : tensor<si64>> : !mo.tensor<[1], si64>
  %shape = mo.broadcast_to<() -> temp>(%one, %shape_shape): (!mo.tensor<[], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[temp], si64>
  %shape_reb = mo.rebind(%shape): !mo.tensor<[temp], si64> -> !mo.tensor<[4], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.reshape([[INPUT]]) : {{.*}} -> !mo.tensor<[1, 1, 1, 1], si64>
  %res = mo.broadcast_to<() -> D0, D1, D2, D3>(%input, %shape_reb) : (
    !mo.tensor<[], si64>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[D0, D1, D2, D3], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output<D0, D1, D2, D3> %res : !mo.tensor<[D0, D1, D2, D3], si64>
}

// CHECK-LABEL: mo.graph @broadcast_to_from_1d_tensor
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>
mo.graph @broadcast_to_from_1d_tensor<() -> O0, O1, O2, O3>(
  %input: !mo.tensor<[], si64>
) -> !mo.tensor<[O0, O1, O2, O3], si64> {

  %one = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[1], si64>
  %shape_shape = mo.constant <#M.dense_array<4> : tensor<si64>> : !mo.tensor<[1], si64>
  %shape = mo.broadcast_to<() -> temp>(%one, %shape_shape): (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[temp], si64>
  %shape_reb = mo.rebind(%shape): !mo.tensor<[temp], si64> -> !mo.tensor<[4], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.reshape([[INPUT]]) : {{.*}} -> !mo.tensor<[1, 1, 1, 1], si64>
  %res = mo.broadcast_to<() -> D0, D1, D2, D3>(%input, %shape_reb) : (
    !mo.tensor<[], si64>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[D0, D1, D2, D3], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output<D0, D1, D2, D3> %res : !mo.tensor<[D0, D1, D2, D3], si64>
}
