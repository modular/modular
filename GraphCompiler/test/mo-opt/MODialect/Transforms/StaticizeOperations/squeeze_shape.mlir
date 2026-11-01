// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @squeeze_shape
//  CHECK-SAME:   <[[N2:.*]], [[N3:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], f32>,
mo.graph @squeeze_shape<N2, N3 -> O1, O2>(
  %input: !mo.tensor<[], f32>,
  %dummy: !mo.tensor<[1, N2, N3, 1], f32>
) -> !mo.tensor<[O1, O2], f32> {

  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[1, N2, N3, 1], f32>) -> !mo.tensor<[4], si32>
  %indices = mo.constant <#M.dense_array<0, -1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %squeezed_shape = mo.squeeze_shape(%dummy_shape, %indices) : (
    !mo.tensor<[4], si32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[2], si32>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[[[N2]], [[N3]]], f32>
  %res = mo.broadcast_to<() -> D0, D1>(%input, %squeezed_shape) : (
    !mo.tensor<[], f32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[D0, D1], f32>

  // CHECK-NEXT: mo.output{{.*}} [[RES]]
  mo.output<D0, D1> %res : !mo.tensor<[D0, D1], f32>
}
