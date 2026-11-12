// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-kernels | FileCheck %s

// A rowwise kernel whose per-row results only a later rowwise kernel with a map
// phase reads, at its own row, runs at the start of that kernel's row. The
// later folds read the earlier folds' results directly.

// Softmax: the max fold runs first, and the sum fold and the map phase read
// it.
// CHECK-LABEL: mo.graph @softmax
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8, 128], f32>)
// CHECK:       %[[R:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  (%[[X]]: !mo.tensor<[8, 128], f32>)
// CHECK-NEXT:  outputs (%[[OUT:.*]]: !mo.tensor<[8, 128], f32>)
// CHECK-NEXT:  row     (%{{.*}}: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
// CHECK-NEXT:    %[[MAX:.*]] = map.iter.reduce @builtin_kernels.monoids.ReduceMax index (%[[I:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[V:.*]] = map.load %[[X]][%[[I]]]
// CHECK-NEXT:      map.yield %[[V]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[SUM:.*]] = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%[[J:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[W:.*]] = map.load %[[X]][%[[J]]]
// CHECK-NEXT:      %[[D:.*]] = map.call.elementwise @builtin_kernels.elementwise.Sub(%[[J]], %[[W]], %[[MAX]])
// CHECK-NEXT:      %[[E:.*]] = map.call.elementwise @builtin_kernels.elementwise.Exp(%[[J]], %[[D]])
// CHECK-NEXT:      map.yield %[[E]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    map.iter.map index (%[[K:.*]]: !map.index<2>) {
// CHECK:           %[[E2:.*]] = map.call.elementwise @builtin_kernels.elementwise.Exp
// CHECK-NEXT:      %[[Y:.*]] = map.call.elementwise @builtin_kernels.elementwise.Div(%[[K]], %[[E2]], %[[SUM]])
// CHECK-NEXT:      map.store %[[OUT]][%[[K]]] = %[[Y]]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NOT:   map.iter.
// CHECK:       mo.output %[[R]]
mo.graph @softmax(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %m = mo.reduce.max(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %mb = mo.static.broadcast_to(%m) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %d = mo.sub(%x, %mb) : !mo.tensor<[8, 128], f32>
  %e = mo.exp(%d) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%e) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %y = mo.div(%e, %sb) : !mo.tensor<[8, 128], f32>
  mo.output %y : !mo.tensor<[8, 128], f32>
}

// A chain: the normalized row feeds the max, which takes it as a producer and
// reads the sum at its own row. One kernel, two folds and a map phase.
// CHECK-LABEL: mo.graph @chainedFolds
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK:         map.iter.reduce @builtin_kernels.monoids.ReduceSum
// CHECK:         map.iter.reduce @builtin_kernels.monoids.ReduceMax
// CHECK:         map.iter.map
// CHECK-NOT:   map.iter.
// CHECK:       mo.output
mo.graph @chainedFolds(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %a = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %ab = mo.static.broadcast_to(%a) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %n = mo.div(%x, %ab) : !mo.tensor<[8, 128], f32>
  %m = mo.reduce.max(%n) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %mb = mo.static.broadcast_to(%m) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %y = mo.sub(%n, %mb) : !mo.tensor<[8, 128], f32>
  mo.output %y : !mo.tensor<[8, 128], f32>
}

// Three dependent folds, each reading the one before at its own row: one
// kernel, the folds in order, then the map phase.
// CHECK-LABEL: mo.graph @threeFolds
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK:         map.iter.reduce @builtin_kernels.monoids.ReduceMax
// CHECK:         map.iter.reduce @builtin_kernels.monoids.ReduceSum
// CHECK:         map.iter.reduce @builtin_kernels.monoids.ReduceMax
// CHECK:         map.iter.map
// CHECK-NOT:   map.iter.
// CHECK:       mo.output
mo.graph @threeFolds(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %m = mo.reduce.max(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %mb = mo.static.broadcast_to(%m) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %d = mo.sub(%x, %mb) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%d) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %n = mo.div(%d, %sb) : !mo.tensor<[8, 128], f32>
  %m2 = mo.reduce.max(%n) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %m2b = mo.static.broadcast_to(%m2) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %y = mo.sub(%n, %m2b) : !mo.tensor<[8, 128], f32>
  mo.output %y : !mo.tensor<[8, 128], f32>
}

// The later kernel writes one value per row, which launches as one phase that
// holds one fold: both kernels stay.
// CHECK-LABEL: mo.graph @twoPassVariance
// CHECK:       %[[A0:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  ({{.*}}, %[[A0]]: !mo.tensor<[8, 1], f32>)
// CHECK-NOT:     map.iter.map
mo.graph @twoPassVariance(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %m = mo.reduce.max(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %mb = mo.static.broadcast_to(%m) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %d = mo.sub(%x, %mb) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%d) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %s : !mo.tensor<[8, 1], f32>
}

// The first result is also a graph output, which needs a side output: both
// kernels stay.
// CHECK-LABEL: mo.graph @firstResultAlsoOutput
// CHECK:       %[[A:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK:       %[[B:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  ({{.*}}, %[[A]]: !mo.tensor<[8, 1], f32>)
// CHECK:       mo.output %[[B]], %[[A]]
mo.graph @firstResultAlsoOutput(%x: !mo.tensor<[8, 128], f32>) -> (!mo.tensor<[8, 128], f32>, !mo.tensor<[8, 1], f32>) {
  %m = mo.reduce.max(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %mb = mo.static.broadcast_to(%m) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %d = mo.sub(%x, %mb) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%d) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %y = mo.div(%d, %sb) : !mo.tensor<[8, 128], f32>
  mo.output %y, %m : !mo.tensor<[8, 128], f32>, !mo.tensor<[8, 1], f32>
}

// Through the transpose, row i reads row j's max: both kernels stay.
// CHECK-LABEL: mo.graph @otherRow(
// CHECK:       %[[A1:.*]] = map.iter.rowwise<[8, 8], axis = 1>
// CHECK:       map.iter.rowwise<[8, 8], axis = 1>
// CHECK-NEXT:  inputs  ({{.*}}, %[[A1]]: !mo.tensor<[8, 1], f32>{{.*}})
mo.graph @otherRow(%x: !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32> {
  %m = mo.reduce.max(%x) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 1], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %mt = mo.transpose(%m, %perm) : (!mo.tensor<[8, 1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 8], f32>
  %mb = mo.static.broadcast_to(%mt) : !mo.tensor<[1, 8], f32> -> !mo.tensor<[8, 8], f32>
  %d = mo.sub(%x, %mb) : !mo.tensor<[8, 8], f32>
  %s = mo.reduce.add(%d) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 8], f32>
  %y = mo.div(%d, %sb) : !mo.tensor<[8, 8], f32>
  mo.output %y : !mo.tensor<[8, 8], f32>
}

// Rows of another length: both kernels stay.
// CHECK-LABEL: mo.graph @otherRowSpace
// CHECK:       %[[A2:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK:       map.iter.rowwise<[8, 64], axis = 1>
// CHECK-NEXT:  inputs  ({{.*}}, %[[A2]]: !mo.tensor<[8, 1], f32>)
mo.graph @otherRowSpace(%x: !mo.tensor<[8, 128], f32>, %z: !mo.tensor<[8, 64], f32>) -> !mo.tensor<[8, 64], f32> {
  %m = mo.reduce.max(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %mb = mo.static.broadcast_to(%m) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 64], f32>
  %d = mo.sub(%z, %mb) : !mo.tensor<[8, 64], f32>
  %s = mo.reduce.add(%d) : (!mo.tensor<[8, 64], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 64], f32>
  %y = mo.div(%d, %sb) : !mo.tensor<[8, 64], f32>
  mo.output %y : !mo.tensor<[8, 64], f32>
}

// The second reduction runs along the other axis: both kernels stay.
// CHECK-LABEL: mo.graph @otherAxis
// CHECK:       %[[A3:.*]] = map.iter.rowwise<[8, 8], axis = 1>
// CHECK:       map.iter.rowwise<[8, 8], axis = 0>
// CHECK-NEXT:  inputs  ({{.*}}, %[[A3]]: !mo.tensor<[8, 1], f32>)
mo.graph @otherAxis(%x: !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32> {
  %m = mo.reduce.max(%x) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 1], f32>
  %mb = mo.static.broadcast_to(%m) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 8], f32>
  %d = mo.sub(%x, %mb) : !mo.tensor<[8, 8], f32>
  %s = mo.reduce.add(%d) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[1, 8], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[1, 8], f32> -> !mo.tensor<[8, 8], f32>
  %y = mo.div(%d, %sb) : !mo.tensor<[8, 8], f32>
  mo.output %y : !mo.tensor<[8, 8], f32>
}
