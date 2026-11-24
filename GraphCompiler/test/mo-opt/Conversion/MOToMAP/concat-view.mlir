// RUN: mo-opt %s --mogg-import-packages --mo-to-map | FileCheck %s

// Verify that mo.concat lowers to a standalone map.iter.elementwise over the output
// shape, not a map.iter.opaque: the body reads each element through a
// map.index.concat.select, one region per input loading that input at the
// region's argument, and stores it at the loop index.

// CHECK-LABEL: mo.graph @concatStatic
// CHECK-NOT:     map.iter.opaque
// CHECK:         map.iter.elementwise<[4, 24]>
// CHECK:         inputs  ([[A:%.*]]: !mo.tensor<[4, 8], f32>, [[B:%.*]]: !mo.tensor<[4, 16], f32>)
// CHECK:         outputs ([[OUT:%.*]]: !mo.tensor<[4, 24], f32>)
// CHECK:         index   ([[IDX:%.*]]: !map.index<2>) -> !mo.tensor<[4, 24], f32> {
// CHECK-NEXT:    [[V:%.*]] = map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 16] -> [4, 24], axis = 1>> ([[IDX]]) : !map.index<2> -> f32 ([[I0:%.*]]: !map.index<2>) {
// CHECK-NEXT:      [[X0:%.*]] = map.load [[A]]{{\[}}[[I0]]] : !mo.tensor<[4, 8], f32>
// CHECK-NEXT:      map.yield [[X0]] : f32
// CHECK-NEXT:    }, ([[I1:%.*]]: !map.index<2>) {
// CHECK-NEXT:      [[X1:%.*]] = map.load [[B]]{{\[}}[[I1]]] : !mo.tensor<[4, 16], f32>
// CHECK-NEXT:      map.yield [[X1]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    map.store [[OUT]]{{\[}}[[IDX]]] = [[V]] : !mo.tensor<[4, 24], f32>
mo.graph @concatStatic(%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 24], f32> {
  %r = mo.concat[1] (%a, %b) : (!mo.tensor<[4, 8], f32>, !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 24], f32>
  mo.output %r : !mo.tensor<[4, 24], f32>
}

// A negative axis is normalized, and each of three inputs gets a region.

// CHECK-LABEL: mo.graph @concatThreeNegativeAxis
// CHECK:         map.index.concat.select<#map.index.concat.spec<[2, 3], [2, 5], [2, 1] -> [2, 9], axis = 1>>
// CHECK-COUNT-3: map.yield
mo.graph @concatThreeNegativeAxis(%a: !mo.tensor<[2, 3], f32>, %b: !mo.tensor<[2, 5], f32>, %c: !mo.tensor<[2, 1], f32>) -> !mo.tensor<[2, 9], f32> {
  %r = mo.concat[-1] (%a, %b, %c) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>, !mo.tensor<[2, 1], f32>) -> !mo.tensor<[2, 9], f32>
  mo.output %r : !mo.tensor<[2, 9], f32>
}

// Symbolic extents stay symbolic in the spec.

// CHECK-LABEL: mo.graph @concatDynamic
// CHECK:         map.iter.elementwise<[M, P]>
// CHECK:         map.index.concat.select<#map.index.concat.spec<[M, N0], [M, N1] -> [M, P], axis = 1>>
mo.graph @concatDynamic<M, N0, N1, P>(%a: !mo.tensor<[M, N0], f32>, %b: !mo.tensor<[M, N1], f32>) -> !mo.tensor<[M, P], f32> {
  %r = mo.concat[1] (%a, %b) : (!mo.tensor<[M, N0], f32>, !mo.tensor<[M, N1], f32>) -> !mo.tensor<[M, P], f32>
  mo.output %r : !mo.tensor<[M, P], f32>
}

// Equal inputs each one wide along the innermost axis interleave: a select
// like any other concat.

// CHECK-LABEL: mo.graph @stackInnermost
// CHECK:         map.iter.elementwise<[64, 2]>
// CHECK:         map.index.concat.select<#map.index.concat.spec<[64, 1], [64, 1] -> [64, 2], axis = 1>>
mo.graph @stackInnermost(%a: !mo.tensor<[64, 1], f32>, %b: !mo.tensor<[64, 1], f32>) -> !mo.tensor<[64, 2], f32> {
  %r = mo.concat[-1] (%a, %b) : (!mo.tensor<[64, 1], f32>, !mo.tensor<[64, 1], f32>) -> !mo.tensor<[64, 2], f32>
  mo.output %r : !mo.tensor<[64, 2], f32>
}
