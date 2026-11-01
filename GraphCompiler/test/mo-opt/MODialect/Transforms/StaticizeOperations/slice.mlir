// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @slice
//  CHECK-SAME:   <[[N2:.*]], [[N3:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], f32>,
mo.graph @slice<N2, N3 -> O1, O2>(
  %input: !mo.tensor<[], f32>,
  %dummy: !mo.tensor<[1, N2, N3, 2], f32>
) -> !mo.tensor<[O1, O2], f32> {

  // This is effectively doing `dummy.shape[-2:]`
  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[1, N2, N3, 2], f32>) -> !mo.tensor<[4], si32>
  %start = mo.constant <#M.dense_array<-2> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %stop = mo.constant <#M.dense_array<100> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %sliced_shape = mo.slice(%dummy_shape, %start, %stop, %step) : (
    !mo.tensor<[4], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>
  ) -> !mo.tensor<[2], si32>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[[[N3]], 2], f32>
  %res = mo.broadcast_to<() -> D0, D1>(%input, %sliced_shape) : (
    !mo.tensor<[], f32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[D0, D1], f32>

  // CHECK-NEXT: mo.output{{.*}} [[RES]]
  mo.output<D0, D1> %res : !mo.tensor<[D0, D1], f32>
}
