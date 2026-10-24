// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @reduce_mul
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]] -> {{.*}}>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>,
mo.graph @reduce_mul<N1, N2 -> out>(
  %input: !mo.tensor<[], si64>,
  %dummy: !mo.tensor<[N1, 10, N2], f32>
) -> !mo.tensor<[out], si64> {

  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, 10, N2], f32>) -> !mo.tensor<[3], si64>
  %shape = mo.reduce.mul(%dummy_shape) [axis = 0] : (
    !mo.tensor<[3], si64>
  ) -> !mo.tensor<[1], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[mul_no_wrap([[N1]], [[N2]], 10)], si64>
  %res = mo.broadcast_to<() -> D0>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[D0], si64>

  // CHECK-NEXT: mo.output{{.*}} [[RES]]
  mo.output<D0> %res : !mo.tensor<[D0], si64>
}
