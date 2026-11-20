// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s


// CHECK-LABEL: mo.graph @reshape
//  CHECK-SAME:   <[[N:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>,
mo.graph @reshape<N -> out>(
  %input: !mo.tensor<[], si64>,
  %dummy: !mo.tensor<[N], f32>
) -> !mo.tensor<[out], si64> {

  %n = mosh.param.to_value = <N>
  %n_scalar = mo.index.to_tensor(%n) -> !mo.tensor<[], si64>
  %one = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %shape = mo.reshape(%n_scalar, %one) : (!mo.tensor<[], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[[[N]]], si64>
  %res = mo.broadcast_to<() -> D0>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[D0], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output<D0> %res : !mo.tensor<[D0], si64>
}
