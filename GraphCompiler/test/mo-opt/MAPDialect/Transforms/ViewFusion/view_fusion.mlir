// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages --mo-to-map --map-fuse-views | FileCheck %s

// ViewFuser is generic over IndexTransformOpInterface: it does not special-case
// slice/broadcast/transpose, and it does not require its producer and consumer
// to share a shape/index space the way ElementwiseFuser does -- that's what
// lets a view fuse into a consumer of a different shape.

// A slice feeding a transpose of a different shape: the transpose's own
// index-transform result (not the raw shared index) becomes the slice's loop
// index, composing the two transforms in the right order.

// CHECK-LABEL: mo.graph @sliceIntoTranspose
// CHECK:       %{{.*}} = map.iter.elementwise<[32, 3]>
// CHECK:       index   (%[[IDX:.*]]: !map.index<2>) -> !mo.tensor<[32, 3], f32> {
// CHECK-NEXT:  %[[TIDX:.*]] = map.index.transpose<{{.*}}> (%[[IDX]], %{{.*}})
// CHECK-NEXT:  %[[SIDX:.*]] = map.index.slice<{{.*}}> (%[[TIDX]], %{{.*}})
// CHECK-NEXT:  %[[VAL:.*]] = map.load %{{.*}}[%[[SIDX]]]
// CHECK-NEXT:  map.store %{{.*}}[%[[IDX]]] = %[[VAL]]
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_view<[#mogg.fusion_leaf<"mo.slice"{{.*}}, #mogg.fusion_leaf<"mo.transpose"{{.*}}]>>
// CHECK-NOT:   map.iter.elementwise
mo.graph @sliceIntoTranspose(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[32, 3], f32> {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %sliced = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.transpose(%sliced, %perm) : (!mo.tensor<[3, 32], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[32, 3], f32>
  mo.output %res : !mo.tensor<[32, 3], f32>
}

// A slice feeding a plain elementwise add: no transform on the consumer
// side, so the slice's own index-transform reads the add's raw shared index
// directly -- the trivial case of the same general mechanism.

// CHECK-LABEL: mo.graph @sliceIntoElementwise
// CHECK:       %{{.*}} = map.iter.elementwise<[3, 32]>
// CHECK:       index   (%[[IDX2:.*]]: !map.index<2>) -> !mo.tensor<[3, 32], f32> {
// CHECK-NEXT:  %[[SIDX2:.*]] = map.index.slice<{{.*}}> (%[[IDX2]], %{{.*}})
// CHECK-NEXT:  %[[SV:.*]] = map.load %{{.*}}[%[[SIDX2]]]
// CHECK-NEXT:  %[[OV:.*]] = map.load %{{.*}}[%[[IDX2]]]
// CHECK-NEXT:  %[[ADD:.*]] = map.call.elementwise @{{.*}}.Add(%[[IDX2]], %[[SV]], %[[OV]])
// CHECK-NEXT:  map.store
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_view<[#mogg.fusion_leaf<"mo.slice"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>>
// CHECK-NOT:   map.iter.elementwise
mo.graph @sliceIntoElementwise(%input: !mo.tensor<[10, 64], f32>, %other: !mo.tensor<[3, 32], f32>) -> !mo.tensor<[3, 32], f32> {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %sliced = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  %res = mo.add(%sliced, %other) : !mo.tensor<[3, 32], f32>
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// A three-hop chain (slice -> transpose -> broadcast) collapses into a
// single kernel across repeated rounds: each round fuses the link nearest
// the top, exposing the next, until nothing more fuses.

// CHECK-LABEL: mo.graph @chainedViews
// CHECK:       %{{.*}} = map.iter.elementwise<[1, 32, 3]>
// CHECK:       index   (%[[IDX3:.*]]: !map.index<3>) -> !mo.tensor<[1, 32, 3], f32> {
// CHECK-NEXT:  %[[BIDX:.*]] = map.index.broadcast<{{.*}}> (%[[IDX3]])
// CHECK-NEXT:  %[[TIDX2:.*]] = map.index.transpose<{{.*}}> (%[[BIDX]], %{{.*}})
// CHECK-NEXT:  %[[SIDX3:.*]] = map.index.slice<{{.*}}> (%[[TIDX2]], %{{.*}})
// CHECK-NEXT:  %[[VAL3:.*]] = map.load %{{.*}}[%[[SIDX3]]]
// CHECK-NEXT:  map.store %{{.*}}[%[[IDX3]]] = %[[VAL3]]
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_view<[#mogg.fusion_leaf<"mo.slice"{{.*}}, #mogg.fusion_leaf<"mo.transpose"{{.*}}, #mogg.fusion_leaf<"mo.static.broadcast_to"{{.*}}]>>
// CHECK-NOT:   map.iter.elementwise
mo.graph @chainedViews(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[1, 32, 3], f32> {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %sliced = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %transposed = mo.transpose(%sliced, %perm) : (!mo.tensor<[3, 32], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[32, 3], f32>
  %res = mo.static.broadcast_to(%transposed) : !mo.tensor<[32, 3], f32> -> !mo.tensor<[1, 32, 3], f32>
  mo.output %res : !mo.tensor<[1, 32, 3], f32>
}

// A view used twice by the same consumer (mo.add(slice, slice)) gets an
// independent copy fused into each use site, mirroring ElementwiseFuser's
// own convention for a duplicated operand.

// CHECK-LABEL: mo.graph @sliceUsedTwice
// CHECK:       %{{.*}} = map.iter.elementwise<[3, 32]>
// CHECK:       index   (%[[IDX4:.*]]: !map.index<2>) -> !mo.tensor<[3, 32], f32> {
// CHECK-NEXT:  %[[SIDX4A:.*]] = map.index.slice<{{.*}}> (%[[IDX4]], %{{.*}})
// CHECK-NEXT:  %[[VAL4A:.*]] = map.load %{{.*}}[%[[SIDX4A]]]
// CHECK-NEXT:  %[[SIDX4B:.*]] = map.index.slice<{{.*}}> (%[[IDX4]], %{{.*}})
// CHECK-NEXT:  %[[VAL4B:.*]] = map.load %{{.*}}[%[[SIDX4B]]]
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Add(%[[IDX4]], %[[VAL4A]], %[[VAL4B]])
// CHECK-NEXT:  map.store
// CHECK-NOT:   map.iter.elementwise
mo.graph @sliceUsedTwice(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[3, 32], f32> {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %sliced = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  %res = mo.add(%sliced, %sliced) : !mo.tensor<[3, 32], f32>
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// A view feeding two DIFFERENT consumers (fan-out, not just a duplicated
// operand of the same consumer) still fuses into both: a view can always be
// duplicated (canDuplicate is unconditionally true), unlike ElementwiseFuser,
// which gates duplication on a tensor-size cost model since it duplicates
// real compute. No standalone slice kernel survives.

// CHECK-LABEL: mo.graph @sliceFeedsTwoConsumers
// CHECK:       %[[R1:.*]] = map.iter.elementwise<[3, 32]>
// CHECK:       index   (%[[IDX6:.*]]: !map.index<2>) -> !mo.tensor<[3, 32], f32> {
// CHECK-NEXT:  %[[SIDX6A:.*]] = map.index.slice<{{.*}}> (%[[IDX6]], %{{.*}})
// CHECK-NEXT:  %[[SV6A:.*]] = map.load %{{.*}}[%[[SIDX6A]]]
// CHECK-NEXT:  %[[YV:.*]] = map.load %{{.*}}[%[[IDX6]]]
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Add(%[[IDX6]], %[[SV6A]], %[[YV]])
// CHECK-NEXT:  map.store
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_view<[#mogg.fusion_leaf<"mo.slice"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>>
// CHECK:       %[[R2:.*]] = map.iter.elementwise<[3, 32]>
// CHECK:       index   (%[[IDX7:.*]]: !map.index<2>) -> !mo.tensor<[3, 32], f32> {
// CHECK-NEXT:  %[[SIDX6B:.*]] = map.index.slice<{{.*}}> (%[[IDX7]], %{{.*}})
// CHECK-NEXT:  %[[SV6B:.*]] = map.load %{{.*}}[%[[SIDX6B]]]
// CHECK-NEXT:  %[[ZV:.*]] = map.load %{{.*}}[%[[IDX7]]]
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Mul(%[[IDX7]], %[[SV6B]], %[[ZV]])
// CHECK-NEXT:  map.store
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_view<[#mogg.fusion_leaf<"mo.slice"{{.*}}, #mogg.fusion_leaf<"mo.mul"{{.*}}]>>
// CHECK:       mo.output %[[R1]], %[[R2]]
mo.graph @sliceFeedsTwoConsumers(%input: !mo.tensor<[10, 64], f32>, %y: !mo.tensor<[3, 32], f32>, %z: !mo.tensor<[3, 32], f32>) -> (!mo.tensor<[3, 32], f32>, !mo.tensor<[3, 32], f32>) {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %sliced = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  %r1 = mo.add(%sliced, %y) : !mo.tensor<[3, 32], f32>
  %r2 = mo.mul(%sliced, %z) : !mo.tensor<[3, 32], f32>
  mo.output %r1, %r2 : !mo.tensor<[3, 32], f32>, !mo.tensor<[3, 32], f32>
}

// ViewFuser alone leaves a pure elementwise chain (no view anywhere)
// completely untouched -- that edge is ElementwiseFuser's job exclusively.

// CHECK-LABEL: mo.graph @pureElementwiseChain
// CHECK:       map.iter.elementwise<[3, 32]>
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.iter.elementwise<[3, 32]>
// CHECK:       map.call.elementwise @{{.*}}.ReLU
mo.graph @pureElementwiseChain(%x: !mo.tensor<[3, 32], f32>, %y: !mo.tensor<[3, 32], f32>) -> !mo.tensor<[3, 32], f32> {
  %added = mo.add(%x, %y) : !mo.tensor<[3, 32], f32>
  %res = mo.relu(%added) : !mo.tensor<[3, 32], f32>
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// A reshape is never a fusion destination: unlike slice/broadcast/transpose,
// composing a reshape with another transform isn't safe to do arbitrarily
// yet, so a slice feeding a reshape stays two separate kernels.

// CHECK-LABEL: mo.graph @sliceIntoReshapeStaysUnfused
// CHECK:       [[SLICED:%.*]] = map.iter.elementwise<[3, 32]>
// CHECK:       map.index.slice<{{.*}}>
// CHECK:       map.iter.elementwise<[96]>
// CHECK:       inputs  ([[SLICED]]: !mo.tensor<[3, 32], f32>)
// CHECK:       map.index.reshape<{{.*}}>
mo.graph @sliceIntoReshapeStaysUnfused(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[96], f32> {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %sliced = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  %res = mo.static.reshape(%sliced) : !mo.tensor<[3, 32], f32> -> !mo.tensor<[96], f32>
  mo.output %res : !mo.tensor<[96], f32>
}

// A view whose result is also a graph output is written to memory either way,
// so it stays its own kernel rather than being re-derived in the consumer.

// CHECK-LABEL: mo.graph @sliceEscapingOutput
// CHECK:       [[SLICED:%.*]] = map.iter.elementwise<[3, 32]>
// CHECK:       map.index.slice<{{.*}}>
// CHECK:       map.iter.elementwise<[3, 32]>
// CHECK:       inputs  ([[SLICED]]: !mo.tensor<[3, 32], f32>
// CHECK-NOT:   map.index.slice<{{.*}}>
// CHECK:       map.call.elementwise @{{.*}}.Add
mo.graph @sliceEscapingOutput(%input: !mo.tensor<[10, 64], f32>, %other: !mo.tensor<[3, 32], f32>) -> (!mo.tensor<[3, 32], f32>, !mo.tensor<[3, 32], f32>) {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %sliced = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  %res = mo.add(%sliced, %other) : !mo.tensor<[3, 32], f32>
  mo.output %sliced, %res : !mo.tensor<[3, 32], f32>, !mo.tensor<[3, 32], f32>
}
