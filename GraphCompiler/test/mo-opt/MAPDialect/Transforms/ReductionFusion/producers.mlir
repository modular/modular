// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-reductions | FileCheck %s

// A producer only a rowwise reads is cloned into the fold at the load's index,
// and the fold reads the producer's inputs instead.

// A chain: `mul` fuses first, which leaves `cast` read only by the rowwise.
// CHECK-LABEL: mo.graph @chain
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8, 128], bf16>)
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  (%[[X]]: !mo.tensor<[8, 128], bf16>)
// CHECK:       map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%[[I:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:    %[[V:.*]] = map.load %[[X]][%[[I]]] : !mo.tensor<[8, 128], bf16>
// CHECK-NEXT:    %[[A:.*]] = map.call.elementwise @builtin_kernels.elementwise.Cast(%[[I]], %[[V]])
// CHECK-NEXT:    %[[SQ:.*]] = map.call.elementwise @builtin_kernels.elementwise.Mul(%[[I]], %[[A]], %[[A]])
// CHECK-NEXT:    map.yield %[[SQ]] : f32
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_rowwise<[#mogg.fusion_leaf<"mo.cast"{{.*}}, #mogg.fusion_leaf<"mo.mul"{{.*}}, #mogg.fusion_leaf<"mo.reduce.add"
mo.graph @chain(%x: !mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 1], f32> {
  %a = mo.cast(%x) : (!mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32>
  %sq = mo.mul(%a, %a) : !mo.tensor<[8, 128], f32>
  %r = mo.reduce.add(%sq) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// A view producer: the broadcast index lands inside the fold.
// CHECK-LABEL: mo.graph @broadcastGamma
// CHECK-SAME:  (%[[X2:.*]]: !mo.tensor<[8, 128], f32>, %[[G:.*]]: !mo.tensor<[128], f32>)
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.iter.reduce @builtin_kernels.monoids.ReduceMax index (%[[I2:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:    %[[XV:.*]] = map.load %[[X2]][%[[I2]]]
// CHECK-NEXT:    %[[GI:.*]] = map.index.broadcast<#map.index.broadcast.spec<[128] -> [8, 128]>> (%[[I2]])
// CHECK-NEXT:    %[[GV:.*]] = map.load %[[G]][%[[GI]]] : !mo.tensor<[128], f32>
// CHECK-NEXT:    map.call.elementwise @builtin_kernels.elementwise.Mul(%[[I2]], %[[XV]], %[[GV]])
mo.graph @broadcastGamma(%x: !mo.tensor<[8, 128], f32>, %g: !mo.tensor<[128], f32>) -> !mo.tensor<[8, 1], f32> {
  %gb = mo.static.broadcast_to(%g) : !mo.tensor<[128], f32> -> !mo.tensor<[8, 128], f32>
  %xg = mo.mul(%x, %gb) : !mo.tensor<[8, 128], f32>
  %r = mo.reduce.max(%xg) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// A non-inner axis.
// CHECK-LABEL: mo.graph @outerAxis
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.iter.rowwise<[M, N], axis = 0>
// CHECK:       map.iter.reduce @builtin_kernels.monoids.ReduceSum
// CHECK:       map.call.elementwise @builtin_kernels.elementwise.Exp
mo.graph @outerAxis<M, N>(%x: !mo.tensor<[M, N], f32>) -> !mo.tensor<[1, N], f32> {
  %e = mo.exp(%x) : !mo.tensor<[M, N], f32>
  %r = mo.reduce.add(%e) : (!mo.tensor<[M, N], f32>) -> !mo.tensor<[1, N], f32>
  mo.output %r : !mo.tensor<[1, N], f32>
}

// A producer another kernel also reads stays materialized.
// CHECK-LABEL: mo.graph @sharedProducer
// CHECK:       %[[SQ2:.*]] = map.iter.elementwise<[8, 128]>
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  (%[[SQ2]]: !mo.tensor<[8, 128], f32>)
mo.graph @sharedProducer(%x: !mo.tensor<[8, 128], f32>) -> (!mo.tensor<[8, 1], f32>, !mo.tensor<[8, 128], f32>) {
  %sq = mo.mul(%x, %x) : !mo.tensor<[8, 128], f32>
  %r = mo.reduce.add(%sq) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %r, %sq : !mo.tensor<[8, 1], f32>, !mo.tensor<[8, 128], f32>
}

// Reshape barrier, as in input fusion: once the reshape fuses, the fold's
// load crosses it, so the slice above it stays its own kernel and writes a
// contiguous tensor.
// CHECK-LABEL: mo.graph @sliceThenReshape
// CHECK:       %[[S:.*]] = map.iter.elementwise<[16, 32]>
// CHECK:       map.index.slice
// CHECK:       map.iter.rowwise<[4, 4, 32], axis = 1>
// CHECK-NEXT:  inputs  (%[[S]]: !mo.tensor<[16, 32], f32>)
// CHECK:       map.index.reshape
// CHECK-NOT:   map.index.slice
// CHECK:       mo.output
mo.graph @sliceThenReshape(%x: !mo.tensor<[16, 64], f32>) -> !mo.tensor<[4, 1, 32], f32> {
  %start = mo.constant <#M.dense_array<0, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<16, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %s = mo.slice(%x, %start, %stop, %step) : (!mo.tensor<[16, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[16, 32], f32>
  %g = mo.static.reshape(%s) : !mo.tensor<[16, 32], f32> -> !mo.tensor<[4, 4, 32], f32>
  %r = mo.reduce.add(%g) : (!mo.tensor<[4, 4, 32], f32>) -> !mo.tensor<[4, 1, 32], f32>
  mo.output %r : !mo.tensor<[4, 1, 32], f32>
}
