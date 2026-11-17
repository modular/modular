// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-kernels | FileCheck %s

// A consumer over the rowwise shape fuses as a map phase after the fold when
// it reads per-row results through a broadcast; a consumer of the map phase's
// result at its own element joins that phase.

// The scalar chain stays at row level; the scale by the row's rsqrt and by
// gamma becomes the map phase. One kernel remains.
// CHECK-LABEL: mo.graph @rmsNorm
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8, 128], f32>, %[[G:.*]]: !mo.tensor<[128], f32>)
// CHECK:       %[[R:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  (%[[X]]: !mo.tensor<[8, 128], f32>, %{{.*}}: !mo.tensor<[], f32>, %[[G]]: !mo.tensor<[128], f32>)
// CHECK-NEXT:  outputs (%[[OUT:.*]]: !mo.tensor<[8, 128], f32>)
// CHECK-NEXT:  row     (%[[ROW:.*]]: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
// CHECK-NEXT:    %[[ACC:.*]] = map.iter.reduce @builtin_kernels.monoids.ReduceSum
// CHECK:         %[[MS:.*]] = map.call.elementwise @builtin_kernels.elementwise.Div(%[[ROW]], %[[ACC]], %{{.*}})
// CHECK-NEXT:    %[[INV:.*]] = map.call.elementwise @builtin_kernels.elementwise.Rsqrt(%[[ROW]], %[[MS]])
// CHECK-NEXT:    map.iter.map index (%[[I:.*]]: !map.index<2>) {
// CHECK-NEXT:      %[[XV:.*]] = map.load %[[X]][%[[I]]] : !mo.tensor<[8, 128], f32>
// CHECK-NEXT:      %[[NX:.*]] = map.call.elementwise @builtin_kernels.elementwise.Mul(%[[I]], %[[XV]], %[[INV]])
// CHECK-NEXT:      %[[GI:.*]] = map.index.broadcast<#map.index.broadcast.spec<[128] -> [8, 128]>> (%[[I]])
// CHECK-NEXT:      %[[GV:.*]] = map.load %[[G]][%[[GI]]] : !mo.tensor<[128], f32>
// CHECK-NEXT:      %[[Y:.*]] = map.call.elementwise @builtin_kernels.elementwise.Mul(%[[I]], %[[NX]], %[[GV]])
// CHECK-NEXT:      map.store %[[OUT]][%[[I]]] = %[[Y]] : !mo.tensor<[8, 128], f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[R]]
mo.graph @rmsNorm(%x: !mo.tensor<[8, 128], f32>, %g: !mo.tensor<[128], f32>) -> !mo.tensor<[8, 128], f32> {
  %sq = mo.mul(%x, %x) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%sq) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %n = mo.constant <#M.dense_array<128.0> : tensor<f32>> : !mo.tensor<[], f32>
  %nb = mo.static.broadcast_to(%n) : !mo.tensor<[], f32> -> !mo.tensor<[8, 1], f32>
  %ms = mo.div(%s, %nb) : !mo.tensor<[8, 1], f32>
  %inv = mo.rsqrt(%ms) : !mo.tensor<[8, 1], f32>
  %invb = mo.static.broadcast_to(%inv) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %nx = mo.mul(%x, %invb) : !mo.tensor<[8, 128], f32>
  %gb = mo.static.broadcast_to(%g) : !mo.tensor<[128], f32> -> !mo.tensor<[8, 128], f32>
  %y = mo.mul(%nx, %gb) : !mo.tensor<[8, 128], f32>
  mo.output %y : !mo.tensor<[8, 128], f32>
}

// A squeezed result, reshaped to [8] and broadcast back over axis 0, reads the
// loop's row too.
// CHECK-LABEL: mo.graph @squeezedOuter
// CHECK:       map.iter.rowwise<[128, 8], axis = 0>
// CHECK:         %[[ACC2:.*]] = map.iter.reduce
// CHECK:         map.iter.map index (%[[I2:.*]]: !map.index<2>) {
// CHECK-NEXT:      %[[XV2:.*]] = map.load %{{.*}}[%[[I2]]]
// CHECK-NEXT:      %[[Y2:.*]] = map.call.elementwise @builtin_kernels.elementwise.Div(%[[I2]], %[[XV2]], %[[ACC2]])
// CHECK-NEXT:      map.store %{{.*}}[%[[I2]]] = %[[Y2]]
// CHECK-NOT:   map.iter.elementwise
mo.graph @squeezedOuter(%x: !mo.tensor<[128, 8], f32>) -> !mo.tensor<[128, 8], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[128, 8], f32>) -> !mo.tensor<[1, 8], f32>
  %sq = mo.static.reshape(%s) : !mo.tensor<[1, 8], f32> -> !mo.tensor<[8], f32>
  %sb = mo.static.broadcast_to(%sq) : !mo.tensor<[8], f32> -> !mo.tensor<[128, 8], f32>
  %y = mo.div(%x, %sb) : !mo.tensor<[128, 8], f32>
  mo.output %y : !mo.tensor<[128, 8], f32>
}

// A symbolic non-inner axis.
// CHECK-LABEL: mo.graph @outerAxis
// CHECK:       map.iter.rowwise<[M, N], axis = 0>
// CHECK:         %[[ACC3:.*]] = map.iter.reduce
// CHECK:         map.iter.map index
// CHECK:           map.call.elementwise @builtin_kernels.elementwise.Div({{.*}}, %[[ACC3]])
// CHECK-NOT:   map.iter.elementwise
mo.graph @outerAxis<M, N>(%x: !mo.tensor<[M, N], f32>) -> !mo.tensor<[M, N], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[M, N], f32>) -> !mo.tensor<[1, N], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[1, N], f32> -> !mo.tensor<[M, N], f32>
  %y = mo.div(%x, %sb) : !mo.tensor<[M, N], f32>
  mo.output %y : !mo.tensor<[M, N], f32>
}

// The exp reads the phase's result at its own element, so it joins the phase;
// the division's output goes away.
// CHECK-LABEL: mo.graph @epilogue
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK:       outputs (%[[OUT4:.*]]: !mo.tensor<[8, 128], f32>)
// CHECK:         map.iter.map index (%[[I4:.*]]: !map.index<2>) {
// CHECK:           %[[D4:.*]] = map.call.elementwise @builtin_kernels.elementwise.Div
// CHECK-NEXT:      %[[E4:.*]] = map.call.elementwise @builtin_kernels.elementwise.Exp(%[[I4]], %[[D4]])
// CHECK-NEXT:      map.store %[[OUT4]][%[[I4]]] = %[[E4]]
// CHECK-NEXT:    }
// CHECK-NOT:   map.iter.elementwise
mo.graph @epilogue(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %y = mo.div(%x, %sb) : !mo.tensor<[8, 128], f32>
  %z = mo.exp(%y) : !mo.tensor<[8, 128], f32>
  mo.output %z : !mo.tensor<[8, 128], f32>
}

// The sum is also a graph output, and a per-row store can't sit next to a map
// phase yet: the division stays its own kernel.
// CHECK-LABEL: mo.graph @perRowOutputKept
// CHECK:       map.iter.rowwise
// CHECK-NOT:     map.iter.map
// CHECK:       map.iter.elementwise<[8, 128]>
mo.graph @perRowOutputKept(%x: !mo.tensor<[8, 128], f32>) -> (!mo.tensor<[8, 128], f32>, !mo.tensor<[8, 1], f32>) {
  %s = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %y = mo.div(%x, %sb) : !mo.tensor<[8, 128], f32>
  mo.output %y, %s : !mo.tensor<[8, 128], f32>, !mo.tensor<[8, 1], f32>
}

// Through the transpose, element (i, j) reads row j's sum: not fused.
// CHECK-LABEL: mo.graph @otherRow
// CHECK:       map.iter.rowwise
// CHECK-NOT:     map.iter.map
// CHECK:       map.iter.elementwise<[8, 8]>
mo.graph @otherRow(%x: !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 1], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %st = mo.transpose(%s, %perm) : (!mo.tensor<[8, 1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 8], f32>
  %sb = mo.static.broadcast_to(%st) : !mo.tensor<[1, 8], f32> -> !mo.tensor<[8, 8], f32>
  %y = mo.div(%x, %sb) : !mo.tensor<[8, 8], f32>
  mo.output %y : !mo.tensor<[8, 8], f32>
}

// The phase's result is also a graph output: it stays, next to the exp's.
// CHECK-LABEL: mo.graph @epilogueSideOutput
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK:       outputs (%[[OD:.*]]: !mo.tensor<[8, 128], f32>, %[[OE:.*]]: !mo.tensor<[8, 128], f32>)
// CHECK:         map.iter.map index (%[[I6:.*]]: !map.index<2>) {
// CHECK:           %[[D6:.*]] = map.call.elementwise @builtin_kernels.elementwise.Div
// CHECK-NEXT:      map.store %[[OD]][%[[I6]]] = %[[D6]]
// CHECK-NEXT:      %[[E6:.*]] = map.call.elementwise @builtin_kernels.elementwise.Exp(%[[I6]], %[[D6]])
// CHECK-NEXT:      map.store %[[OE]][%[[I6]]] = %[[E6]]
// CHECK-NOT:   map.iter.elementwise
mo.graph @epilogueSideOutput(%x: !mo.tensor<[8, 128], f32>) -> (!mo.tensor<[8, 128], f32>, !mo.tensor<[8, 128], f32>) {
  %s = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %y = mo.div(%x, %sb) : !mo.tensor<[8, 128], f32>
  %z = mo.exp(%y) : !mo.tensor<[8, 128], f32>
  mo.output %z, %y : !mo.tensor<[8, 128], f32>, !mo.tensor<[8, 128], f32>
}

// A two-pass variance: the difference from the mean feeds a second reduction,
// which takes it as a producer. Fusing it into the first reduction instead
// would write it out and read it back.
// CHECK-LABEL: mo.graph @variance
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NOT:     map.iter.map
// CHECK:         map.call.elementwise @builtin_kernels.elementwise.Div
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK:         map.iter.reduce
// CHECK:           map.call.elementwise @builtin_kernels.elementwise.Sub
// CHECK:           map.call.elementwise @builtin_kernels.elementwise.Mul
// CHECK-NOT:   map.iter.elementwise
mo.graph @variance(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %n = mo.constant <#M.dense_array<128.0> : tensor<f32>> : !mo.tensor<[], f32>
  %nb = mo.static.broadcast_to(%n) : !mo.tensor<[], f32> -> !mo.tensor<[8, 1], f32>
  %m = mo.div(%s, %nb) : !mo.tensor<[8, 1], f32>
  %mb = mo.static.broadcast_to(%m) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %d = mo.sub(%x, %mb) : !mo.tensor<[8, 128], f32>
  %d2 = mo.mul(%d, %d) : !mo.tensor<[8, 128], f32>
  %v = mo.reduce.add(%d2) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %v : !mo.tensor<[8, 1], f32>
}

// Two transposes bring the sum back to its own row: it fuses.
// CHECK-LABEL: mo.graph @transposedTwice
// CHECK:       map.iter.rowwise<[8, 8], axis = 1>
// CHECK:         map.iter.map
// CHECK-NOT:   map.iter.elementwise
mo.graph @transposedTwice(%x: !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 1], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %t1 = mo.transpose(%s, %perm) : (!mo.tensor<[8, 1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 8], f32>
  %t2 = mo.transpose(%t1, %perm) : (!mo.tensor<[1, 8], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%t2) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 8], f32>
  %y = mo.div(%x, %sb) : !mo.tensor<[8, 8], f32>
  mo.output %y : !mo.tensor<[8, 8], f32>
}
