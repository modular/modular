// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @cast
//  CHECK-SAME:   <[[N:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>,
mo.graph @cast<N -> out>(
  %input: !mo.tensor<[], si64>,
  %dummy: !mo.tensor<[N], si64>
) -> !mo.tensor<[out], si64> {

  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N], si64>) -> !mo.tensor<[1], si64>
  %shape = mo.cast(%dummy_shape) : (!mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[[[N]]], si64>
  %res = mo.broadcast_to<() -> D0>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[D0], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output<D0> %res : !mo.tensor<[D0], si64>
}
