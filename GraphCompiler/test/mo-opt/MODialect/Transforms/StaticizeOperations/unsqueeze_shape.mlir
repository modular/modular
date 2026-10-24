// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @unsqueeze_shape
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], f32>,
mo.graph @unsqueeze_shape<N1, N2 -> O1, O2, O3, O4, O5>(
  %input: !mo.tensor<[], f32>,
  %dummy: !mo.tensor<[N1, N2], f32>
) -> !mo.tensor<[O1, O2, O3, O4, O5], f32> {

  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, N2], f32>) -> !mo.tensor<[2], si32>
  %indices = mo.constant <#M.dense_array<0, 2, -1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %unsqueezed_shape = mo.unsqueeze_shape(%dummy_shape, %indices) : (
    !mo.tensor<[2], si32>, !mo.tensor<[3], si32>
  ) -> !mo.tensor<[5], si32>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[1, [[N1]], 1, [[N2]], 1], f32>
  %res = mo.broadcast_to<() -> D0, D1, D2, D3, D4>(%input, %unsqueezed_shape) : (
    !mo.tensor<[], f32>, !mo.tensor<[5], si32>
  ) -> !mo.tensor<[D0, D1, D2, D3, D4], f32>

  // CHECK-NEXT: mo.output{{.*}} [[RES]]
  mo.output<D0, D1, D2, D3, D4> %res : !mo.tensor<[D0, D1, D2, D3, D4], f32>
}
