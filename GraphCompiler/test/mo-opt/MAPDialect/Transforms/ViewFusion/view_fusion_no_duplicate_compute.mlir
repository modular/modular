// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-views | FileCheck %s

// A residual stream h1 = h0 + reshape(m1), consumed by both the next residual
// add and an exp. Once the reshape fuses into the add, the add reads through
// an index transform, so it is a view producer -- but it is also compute over
// two large tensors. Duplicating it into both consumers would recompute the
// residual (and keep m1 alive) in each; chained over layers that is quadratic.
// It must be materialized once and shared, as the legacy fuser does.

// CHECK-LABEL: mo.graph @residualThroughReshape
// CHECK:       %[[H1:.*]] = map.iter.elementwise<[2, 64, 256]>
// CHECK:       map.index.reshape
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.iter.elementwise<[2, 64, 256]>
// CHECK-NEXT:  inputs (%[[H1]]: {{.*}})
// CHECK:       map.call.elementwise @{{.*}}.Exp
// CHECK:       map.iter.elementwise<[2, 64, 256]>
// CHECK-NEXT:  inputs (%[[H1]]: {{.*}}, %{{.*}}: {{.*}})
// CHECK:       map.call.elementwise @{{.*}}.Add
mo.graph @residualThroughReshape(%h0: !mo.tensor<[2, 64, 256], f32>, %m1: !mo.tensor<[128, 256], f32>, %m2: !mo.tensor<[128, 256], f32>) -> (!mo.tensor<[2, 64, 256], f32>, !mo.tensor<[2, 64, 256], f32>) {
  %r1 = mo.static.reshape(%m1) : !mo.tensor<[128, 256], f32> -> !mo.tensor<[2, 64, 256], f32>
  %h1 = mo.add(%h0, %r1) : !mo.tensor<[2, 64, 256], f32>
  %e = mo.exp(%h1) : !mo.tensor<[2, 64, 256], f32>
  %r2 = mo.static.reshape(%m2) : !mo.tensor<[128, 256], f32> -> !mo.tensor<[2, 64, 256], f32>
  %h2 = mo.add(%h1, %r2) : !mo.tensor<[2, 64, 256], f32>
  mo.output %e, %h2 : !mo.tensor<[2, 64, 256], f32>, !mo.tensor<[2, 64, 256], f32>
}

// A pure view with the same fan-out is still duplicated: re-deriving an index
// is free, so each consumer reads m through its own reshape.

// CHECK-LABEL: mo.graph @pureViewStillDuplicated
// CHECK:       map.iter.elementwise<[2, 64, 256]>
// CHECK:       map.index.reshape
// CHECK:       map.call.elementwise @{{.*}}.Exp
// CHECK:       map.iter.elementwise<[2, 64, 256]>
// CHECK:       map.index.reshape
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK-NOT:   map.iter.elementwise
mo.graph @pureViewStillDuplicated(%m: !mo.tensor<[128, 256], f32>, %x: !mo.tensor<[2, 64, 256], f32>) -> (!mo.tensor<[2, 64, 256], f32>, !mo.tensor<[2, 64, 256], f32>) {
  %r = mo.static.reshape(%m) : !mo.tensor<[128, 256], f32> -> !mo.tensor<[2, 64, 256], f32>
  %e = mo.exp(%r) : !mo.tensor<[2, 64, 256], f32>
  %a = mo.add(%r, %x) : !mo.tensor<[2, 64, 256], f32>
  mo.output %e, %a : !mo.tensor<[2, 64, 256], f32>, !mo.tensor<[2, 64, 256], f32>
}
