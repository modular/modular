// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-kernels --map-rlicm --map-rcse | FileCheck %s

// The row computes the value every phase computes from the input the first
// fold reduces once per element, in a `map.iter.cache`. Of the candidates, the
// cache takes the narrowest, then the one with the most ops. Each phase's copy
// of it becomes `map.iter.cache.load`.

// CHECK-LABEL: mo.graph @plainInput
// CHECK:         %[[P1:.*]] = map.iter.cache index (%[[K1:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[X1:.*]] = map.load %{{.*}}[%[[K1]]]
// CHECK-NEXT:      map.yield %[[X1]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    map.iter.reduce
// CHECK-NEXT:      %[[V1:.*]] = map.iter.cache.load %[[P1]][%{{.*}}] : !map.index<2>, f32
// CHECK-NEXT:      map.yield %[[V1]] : f32
mo.graph @plainInput(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %s : !mo.tensor<[8, 1], f32>
}

// Both phases compute x / y: the Row loads the quotient.
// CHECK-LABEL: mo.graph @quotient
// CHECK:         %[[P2:.*]] = map.iter.cache index (%[[K2:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[X2:.*]] = map.load %{{.*}}[%[[K2]]]
// CHECK-NEXT:      %[[Y2:.*]] = map.load %{{.*}}[%[[K2]]]
// CHECK-NEXT:      %[[Q2:.*]] = map.call.elementwise @builtin_kernels.elementwise.Div(%[[K2]], %[[X2]], %[[Y2]])
// CHECK-NEXT:      map.yield %[[Q2]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    map.iter.reduce
// CHECK-NEXT:      %[[V2:.*]] = map.iter.cache.load %[[P2]][%{{.*}}] : !map.index<2>, f32
// CHECK-NEXT:      map.yield %[[V2]] : f32
// CHECK:         map.iter.map index (%[[J2:.*]]: !map.index<2>) {
// CHECK-NEXT:      %[[W2:.*]] = map.iter.cache.load %[[P2]][%{{.*}}] : !map.index<2>, f32
// CHECK-NEXT:      map.call.elementwise @builtin_kernels.elementwise.Div(%[[J2]], %[[W2]],
mo.graph @quotient(%x: !mo.tensor<[8, 128], f32>, %y: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %q = mo.div(%x, %y) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%q) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %r = mo.div(%q, %sb) : !mo.tensor<[8, 128], f32>
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// A cast to f32 would double the cached row: the Row loads x in bf16, and
// each phase casts it.
// CHECK-LABEL: mo.graph @narrow
// CHECK:         %[[P3:.*]] = map.iter.cache index (%[[K3:.*]]: !map.index<2>) -> bf16 {
// CHECK-NEXT:      %[[X3:.*]] = map.load %{{.*}}[%[[K3]]]
// CHECK-NEXT:      map.yield %[[X3]] : bf16
// CHECK-NEXT:    }
// CHECK-NEXT:    map.iter.reduce
// CHECK-NEXT:      %[[V3:.*]] = map.iter.cache.load %[[P3]][%{{.*}}] : !map.index<2>, bf16
// CHECK-NEXT:      map.call.elementwise @builtin_kernels.elementwise.Cast({{.*}}, %[[V3]])
// CHECK:         map.iter.map
// CHECK-NEXT:      %[[W3:.*]] = map.iter.cache.load %[[P3]][%{{.*}}] : !map.index<2>, bf16
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
// CHECK:         map.iter.cache index (%[[K4:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[X4:.*]] = map.load %{{.*}}[%[[K4]]]
// CHECK-NEXT:      map.yield %[[X4]] : f32
// CHECK-NEXT:    }
// CHECK:         map.iter.map
// CHECK:           map.iter.cache.load
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
// CHECK:         map.iter.cache index (%[[K5:.*]]: !map.index<2>) -> f32 {
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
// CHECK:         map.iter.cache index (%[[K6:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[X6:.*]] = map.load %{{.*}}[%[[K6]]]
// CHECK-NEXT:      %[[CI6:.*]] = map.index.broadcast{{.*}}(%[[K6]])
// CHECK-NEXT:      %[[C6:.*]] = map.load %{{.*}}[%[[CI6]]]
// CHECK-NEXT:      %[[M6:.*]] = map.call.elementwise @builtin_kernels.elementwise.Mul(%[[K6]], %[[X6]], %[[C6]])
// CHECK-NEXT:      map.yield %[[M6]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    map.iter.reduce
// CHECK-NEXT:      map.iter.cache.load
mo.graph @scaled(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %c = mo.constant <#M.dense_array<2.0> : tensor<f32>> : !mo.tensor<[], f32>
  %cb = mo.static.broadcast_to(%c) : !mo.tensor<[], f32> -> !mo.tensor<[8, 128], f32>
  %m = mo.mul(%x, %cb) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%m) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %s : !mo.tensor<[8, 1], f32>
}
