// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-kernels --map-rlicm --cse --map-rcse | FileCheck %s

// The row computes the value every phase computes from the input the first
// fold reduces once per element, in a `map.iter.tile`. Of the candidates, the
// tile takes the narrowest, then the one with the most ops. Each phase's copy
// of it becomes `map.iter.tile.load`.

// CHECK-LABEL: mo.graph @plainInput
// CHECK:         %[[P1:.*]] = map.iter.tile index (%[[K1:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[X1:.*]] = map.load %{{.*}}[%[[K1]]]
// CHECK-NEXT:      map.yield %[[X1]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    map.iter.reduce
// CHECK-NEXT:      %[[V1:.*]] = map.iter.tile.load %[[P1]][%{{.*}}] : !map.index<2>, f32
// CHECK-NEXT:      map.yield %[[V1]] : f32
mo.graph @plainInput(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %s : !mo.tensor<[8, 1], f32>
}

// Both phases compute x / y: the Row loads the quotient.
// CHECK-LABEL: mo.graph @quotient
// CHECK:         %[[P2:.*]] = map.iter.tile index (%[[K2:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[X2:.*]] = map.load %{{.*}}[%[[K2]]]
// CHECK-NEXT:      %[[Y2:.*]] = map.load %{{.*}}[%[[K2]]]
// CHECK-NEXT:      %[[Q2:.*]] = map.call.elementwise @builtin_kernels.elementwise.Div(%[[K2]], %[[X2]], %[[Y2]])
// CHECK-NEXT:      map.yield %[[Q2]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    map.iter.reduce
// CHECK-NEXT:      %[[V2:.*]] = map.iter.tile.load %[[P2]][%{{.*}}] : !map.index<2>, f32
// CHECK-NEXT:      map.yield %[[V2]] : f32
// CHECK:         map.iter.map index (%[[J2:.*]]: !map.index<2>) {
// CHECK-NEXT:      %[[W2:.*]] = map.iter.tile.load %[[P2]][%{{.*}}] : !map.index<2>, f32
// CHECK-NEXT:      map.call.elementwise @builtin_kernels.elementwise.Div(%[[J2]], %[[W2]],
mo.graph @quotient(%x: !mo.tensor<[8, 128], f32>, %y: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %q = mo.div(%x, %y) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%q) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %r = mo.div(%q, %sb) : !mo.tensor<[8, 128], f32>
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// A cast to f32 would double the tile: the Row loads x in bf16, and
// each phase casts it.
// CHECK-LABEL: mo.graph @narrow
// CHECK:         %[[P3:.*]] = map.iter.tile index (%[[K3:.*]]: !map.index<2>) -> bf16 {
// CHECK-NEXT:      %[[X3:.*]] = map.load %{{.*}}[%[[K3]]]
// CHECK-NEXT:      map.yield %[[X3]] : bf16
// CHECK-NEXT:    }
// CHECK-NEXT:    map.iter.reduce
// CHECK-NEXT:      %[[V3:.*]] = map.iter.tile.load %[[P3]][%{{.*}}] : !map.index<2>, bf16
// CHECK-NEXT:      map.call.elementwise @builtin_kernels.elementwise.Cast({{.*}}, %[[V3]])
// CHECK:         map.iter.map
// CHECK-NEXT:      %[[W3:.*]] = map.iter.tile.load %[[P3]][%{{.*}}] : !map.index<2>, bf16
// CHECK-NEXT:      map.call.elementwise @builtin_kernels.elementwise.Cast({{.*}}, %[[W3]])
mo.graph @narrow(%x: !mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32> {
  %a = mo.cast(%x) : (!mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32>
  %sq = mo.mul(%a, %a) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%sq) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %r = mo.div(%a, %sb) : !mo.tensor<[8, 128], f32>
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// The map phase also reads x by itself, so the Row loads x, and the quotient
// stays in each phase.
// CHECK-LABEL: mo.graph @rawRead
// CHECK:         map.iter.tile index (%[[K4:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[X4:.*]] = map.load %{{.*}}[%[[K4]]]
// CHECK-NEXT:      map.yield %[[X4]] : f32
// CHECK-NEXT:    }
// CHECK:         map.iter.map
// CHECK:           map.iter.tile.load
// CHECK:           map.call.elementwise @builtin_kernels.elementwise.Div
mo.graph @rawRead(%x: !mo.tensor<[8, 128], f32>, %y: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %q = mo.div(%x, %y) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%q) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %t = mo.div(%q, %sb) : !mo.tensor<[8, 128], f32>
  %r = mo.mul(%t, %x) : !mo.tensor<[8, 128], f32>
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// The map phase reads the quotient p outside the larger q = p / y, so the Row
// loads p, not q.
// CHECK-LABEL: mo.graph @intermediateReused
// CHECK:         map.iter.tile index (%[[K5:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[X5:.*]] = map.load %{{.*}}[%[[K5]]]
// CHECK-NEXT:      %[[Y5:.*]] = map.load %{{.*}}[%[[K5]]]
// CHECK-NEXT:      %[[Q5:.*]] = map.call.elementwise @builtin_kernels.elementwise.Div(%[[K5]], %[[X5]], %[[Y5]])
// CHECK-NEXT:      map.yield %[[Q5]] : f32
// CHECK-NEXT:    }
mo.graph @intermediateReused(%x: !mo.tensor<[8, 128], f32>, %y: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %p = mo.div(%x, %y) : !mo.tensor<[8, 128], f32>
  %q = mo.div(%p, %y) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%q) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %t = mo.div(%q, %sb) : !mo.tensor<[8, 128], f32>
  %r = mo.div(%t, %p) : !mo.tensor<[8, 128], f32>
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// A load through a view of the element's index, here a broadcast scalar, is
// per element too: the product moves into the load.
// CHECK-LABEL: mo.graph @scaled
// CHECK:         map.iter.tile index (%[[K6:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[X6:.*]] = map.load %{{.*}}[%[[K6]]]
// CHECK-NEXT:      %[[CI6:.*]] = map.index.broadcast{{.*}}(%[[K6]])
// CHECK-NEXT:      %[[C6:.*]] = map.load %{{.*}}[%[[CI6]]]
// CHECK-NEXT:      %[[M6:.*]] = map.call.elementwise @builtin_kernels.elementwise.Mul(%[[K6]], %[[X6]], %[[C6]])
// CHECK-NEXT:      map.yield %[[M6]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    map.iter.reduce
// CHECK-NEXT:      map.iter.tile.load
mo.graph @scaled(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %c = mo.constant <#M.dense_array<2.0> : tensor<f32>> : !mo.tensor<[], f32>
  %cb = mo.static.broadcast_to(%c) : !mo.tensor<[], f32> -> !mo.tensor<[8, 128], f32>
  %m = mo.mul(%x, %cb) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%m) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %s : !mo.tensor<[8, 1], f32>
}

// Rotate-half after a reduction: the map phase computes x * sum at its element
// and, in the select's regions, at its partner in the other half of the row.
// That value becomes a shared cache, which the regions read 8 elements away.
// CHECK-LABEL: mo.graph @rotateHalfShares
// CHECK:         map.iter.reduce
// CHECK:         %[[C:.*]] = map.iter.cache index (%[[K:.*]]: !map.index<2>) -> f32 {
// CHECK:           map.call.elementwise @builtin_kernels.elementwise.Mul(%[[K]],
// CHECK:         } {shared = true}
// CHECK:         map.iter.map index (%[[J:.*]]: !map.index<2>) {
// CHECK:           map.iter.cache.load %[[C]][%[[J]]] : !map.index<2>, f32
// CHECK:           map.index.concat.select
// CHECK:             map.iter.cache.load %[[C]][%[[J]]] offset(8) : !map.index<2>, f32
// CHECK:             map.iter.cache.load %[[C]][%[[J]]] offset(-8) : !map.index<2>, f32
// CHECK-NOT:     map.load
// CHECK:       mo.output
mo.graph @rotateHalfShares(%x: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 16], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[4, 16], f32>
  %y = mo.mul(%x, %sb) : !mo.tensor<[4, 16], f32>
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<4, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<4, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %y1 = mo.slice(%y, %start1, %stop1, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 8], f32>
  %y2 = mo.slice(%y, %start2, %stop2, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 8], f32>
  %ny2 = mo.negative(%y2) : !mo.tensor<[4, 8], f32>
  %r = mo.concat[1] (%ny2, %y1) : (!mo.tensor<[4, 8], f32>, !mo.tensor<[4, 8], f32>) -> !mo.tensor<[4, 16], f32>
  %o = mo.add(%y, %r) : !mo.tensor<[4, 16], f32>
  mo.output %o : !mo.tensor<[4, 16], f32>
}

// A value the regions compute with no load is cheaper to compute again than
// to share: the row gets no cache.
// CHECK-LABEL: mo.graph @rotateHalfOfRowValue
// CHECK-NOT:     map.iter.cache
// CHECK:       mo.output
mo.graph @rotateHalfOfRowValue(%x: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 16], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[4, 16], f32>
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<4, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<4, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %y1 = mo.slice(%sb, %start1, %stop1, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 8], f32>
  %y2 = mo.slice(%sb, %start2, %stop2, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 8], f32>
  %ny2 = mo.negative(%y2) : !mo.tensor<[4, 8], f32>
  %r = mo.concat[1] (%ny2, %y1) : (!mo.tensor<[4, 8], f32>, !mo.tensor<[4, 8], f32>) -> !mo.tensor<[4, 16], f32>
  %o = mo.mul(%x, %r) : !mo.tensor<[4, 16], f32>
  mo.output %o : !mo.tensor<[4, 16], f32>
}

// Rotate-half of two values, x * sum and x * x: their copies would need two
// caches, and a row has one, so it gets none.
// CHECK-LABEL: mo.graph @rotateHalfOfTwo
// CHECK-NOT:     map.iter.cache
// CHECK:       mo.output
mo.graph @rotateHalfOfTwo(%x: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 16], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[4, 16], f32>
  %y = mo.mul(%x, %sb) : !mo.tensor<[4, 16], f32>
  %z = mo.mul(%x, %x) : !mo.tensor<[4, 16], f32>
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<4, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<4, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %y1 = mo.slice(%y, %start1, %stop1, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 8], f32>
  %y2 = mo.slice(%y, %start2, %stop2, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 8], f32>
  %z1 = mo.slice(%z, %start1, %stop1, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 8], f32>
  %z2 = mo.slice(%z, %start2, %stop2, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 8], f32>
  %ry = mo.concat[1] (%y2, %y1) : (!mo.tensor<[4, 8], f32>, !mo.tensor<[4, 8], f32>) -> !mo.tensor<[4, 16], f32>
  %rz = mo.concat[1] (%z2, %z1) : (!mo.tensor<[4, 8], f32>, !mo.tensor<[4, 8], f32>) -> !mo.tensor<[4, 16], f32>
  %a = mo.add(%y, %ry) : !mo.tensor<[4, 16], f32>
  %b = mo.add(%z, %rz) : !mo.tensor<[4, 16], f32>
  %o = mo.add(%a, %b) : !mo.tensor<[4, 16], f32>
  mo.output %o : !mo.tensor<[4, 16], f32>
}

// A concat over the rows, not along them: the select's regions read x two
// rows away, which no offset along the row reaches, so the row gets no cache.
// CHECK-LABEL: mo.graph @swapRowsGetsNoCache
// CHECK-NOT:     map.iter.cache
// CHECK:         map.iter.map index
// CHECK:           map.index.concat.select<#map.index.concat.spec<[2, 16], [2, 16] -> [4, 16], axis = 0>>
// CHECK:       mo.output
mo.graph @swapRowsGetsNoCache(%x: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 16], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[4, 16], f32>
  %y = mo.mul(%x, %sb) : !mo.tensor<[4, 16], f32>
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<2, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<2, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<4, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %x1 = mo.slice(%x, %start1, %stop1, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 16], f32>
  %x2 = mo.slice(%x, %start2, %stop2, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 16], f32>
  %c = mo.concat[0] (%x2, %x1) : (!mo.tensor<[2, 16], f32>, !mo.tensor<[2, 16], f32>) -> !mo.tensor<[4, 16], f32>
  %o = mo.add(%y, %c) : !mo.tensor<[4, 16], f32>
  mo.output %o : !mo.tensor<[4, 16], f32>
}
