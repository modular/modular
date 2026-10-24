// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-views --map-fuse-horizontal --map-fuse-epilogue | FileCheck %s

// Multi-output consumers that epilogue fusion must leave alone. Horizontal
// fusion builds the consumer, as in `epilogue_fusion_multi_output.mlir`.

// A view between the producer and the pair: view fusion moves the transpose
// into both adds' loads, so the pair reads the matmul at a transposed index,
// which an epilogue store cannot express. The matmul keeps its single store
// and the pair stays its own kernel.
// CHECK-LABEL: mo.graph @transposed_pair
// CHECK:       %[[MM:[0-9]+]] = map.iter.opaque
// CHECK:       map.iter.epilogue
// CHECK-NEXT:  map.store
// CHECK-NEXT:  }
// CHECK:       %[[PAIR:[0-9]+]]:2 = map.iter.elementwise<[8, 8]>
// CHECK:       map.index.transpose
// CHECK-NEXT:  map.load %[[MM]][
// CHECK:       mo.output %[[PAIR]]#0, %[[PAIR]]#1
mo.graph @transposed_pair(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>,
  %w: !mo.tensor<[8, 8], f32>
) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %t = mo.transpose(%mm, %perm) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 8], f32>
  %a = mo.add(%t, %z) : !mo.tensor<[8, 8], f32>
  %b = mo.add(%t, %w) : !mo.tensor<[8, 8], f32>
  mo.output %a, %b : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}

// The pair fuses into the matmul, whose epilogue then stores twice. The relu
// reading the first add's result would need that epilogue rebuilt around one
// of its two stores, which fusion does not do yet (TODO(GEX-4223) in
// `findFusibleEdge`), so the relu stays its own kernel. The relu is the
// first add's only reader: were that add also returned, horizontal fusion
// would merge the add and the relu before the pair reaches the matmul.
// CHECK-LABEL: mo.graph @consumer_of_two_store_epilogue
// CHECK:       %[[MM:[0-9]+]]:2 = map.iter.opaque
// CHECK-COUNT-2: map.store
// CHECK:       map.call.opaque @{{.*}}.Matmul
// CHECK:       %[[RELU:[0-9]+]] = map.iter.elementwise<[8, 8]>
// CHECK-NEXT:  inputs (%[[MM]]#0: !mo.tensor<[8, 8], f32>
// CHECK:       map.call.elementwise @{{.*}}.ReLU
// CHECK:       mo.output %[[RELU]], %[[MM]]#1
mo.graph @consumer_of_two_store_epilogue(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>,
  %w: !mo.tensor<[8, 8], f32>
) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %a = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32>
  %b = mo.add(%mm, %w) : !mo.tensor<[8, 8], f32>
  %r = mo.relu(%a) : !mo.tensor<[8, 8], f32>
  mo.output %r, %b : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}
