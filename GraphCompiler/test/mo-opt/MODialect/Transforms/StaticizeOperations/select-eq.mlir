// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @select_eq_cond
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]], [[N3:.*]], [[N4:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], f32>,
mo.graph @select_eq_cond<N1, N2, N3, N4 -> O1, O2>(
  %input: !mo.tensor<[], f32>,
  %dummy1: !mo.tensor<[N1, N2], f32>,
  %dummy2: !mo.tensor<[N3, N4], f32>
) -> !mo.tensor<[O1, O2], f32> {

  %dummy1_shape = mo.shape_of(%dummy1) : (!mo.tensor<[N1, N2], f32>) -> !mo.tensor<[2], si32>
  %dummy2_shape = mo.shape_of(%dummy2) : (!mo.tensor<[N3, N4], f32>) -> !mo.tensor<[2], si32>
  %neg_ones = mo.constant <#M.dense_array<-1, -1> : !M.array<2xsi32>> : !mo.tensor<[2], si32>
  // NOTE constant folding won't be able to fold this, we'll need the range
  // analysis from `PropagateShapesPass`.
  %cond = mo.equal(%dummy1_shape, %neg_ones) : (
    !mo.tensor<[2], si32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[2], bool>
  %shape = mo.select(%cond, %dummy1_shape, %dummy2_shape) : (
    !mo.tensor<[2], bool>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[2], si32>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[[[N3]], [[N4]]], f32>
  %res = mo.broadcast_to<() -> D0, D1>(%input, %shape) : (
    !mo.tensor<[], f32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[D0, D1], f32>

  // CHECK-NEXT: mo.output{{.*}} [[RES]]
  mo.output<D0, D1> %res : !mo.tensor<[D0, D1], f32>
}
