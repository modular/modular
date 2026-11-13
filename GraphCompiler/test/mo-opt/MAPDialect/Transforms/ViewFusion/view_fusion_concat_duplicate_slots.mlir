// RUN: mo-opt --mlir-print-local-scope %s --map-fuse-views | FileCheck %s

// A consumer that takes the concat through two input slots reads it at its own
// index through one and at a transposed index through the other. The select
// can only run at a loop index, so the concat must not fuse: both slots are
// one tensor, and the transposed read rules it out. Hand-written MAP; the
// graph path never produces duplicate slots.

// CHECK-LABEL: mo.graph @concatThroughTwoSlots
// CHECK:       %[[C:.*]] = map.iter.elementwise<[8, 8]>
// CHECK:       map.index.concat.select
// CHECK:       map.iter.elementwise<[8, 8]>
// CHECK-NEXT:  inputs (%[[C]]
// CHECK-NOT:   map.index.concat.select
// CHECK:       map.index.transpose
// CHECK:       mo.output
mo.graph @concatThroughTwoSlots(%a: !mo.tensor<[8, 4], f32>, %b: !mo.tensor<[8, 4], f32>, %perm: !mo.tensor<[2], si64>) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %c = map.iter.elementwise<[8, 8]>
      inputs  (%a: !mo.tensor<[8, 4], f32>, %b: !mo.tensor<[8, 4], f32>)
      outputs (%o: !mo.tensor<[8, 8], f32>)
      index   (%i: !map.index<2>) -> !mo.tensor<[8, 8], f32> {
    %v = map.index.concat.select<#map.index.concat.spec<[8, 4], [8, 4] -> [8, 8], axis = 1>> (%i) : !map.index<2> -> f32 (%k: !map.index<2>) {
      %xa = map.load %a[%k] : !mo.tensor<[8, 4], f32>
      map.yield %xa : f32
    }, (%k: !map.index<2>) {
      %yb = map.load %b[%k] : !mo.tensor<[8, 4], f32>
      map.yield %yb : f32
    }
    map.store %o[%i] = %v : !mo.tensor<[8, 8], f32>
  }
  %r:2 = map.iter.elementwise<[8, 8]>
      inputs  (%c as %s0: !mo.tensor<[8, 8], f32>, %c as %s1: !mo.tensor<[8, 8], f32>, %perm as %p: !mo.tensor<[2], si64>)
      outputs (%o0: !mo.tensor<[8, 8], f32>, %o1: !mo.tensor<[8, 8], f32>)
      index   (%j: !map.index<2>) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
    %u = map.load %s0[%j] : !mo.tensor<[8, 8], f32>
    %jt = map.index.transpose<#map.index.transpose.spec<[8, 8] -> [8, 8], perm = [1, 0]>>(%j, %p)
        : (!map.index<2>, !mo.tensor<[2], si64>) -> !map.index<2>
    %w = map.load %s1[%jt] : !mo.tensor<[8, 8], f32>
    map.store %o0[%j] = %u : !mo.tensor<[8, 8], f32>
    map.store %o1[%j] = %w : !mo.tensor<[8, 8], f32>
  }
  mo.output %r#0, %r#1 : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}
