// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// Checks that kernel fusion fuses every reduction pattern we care about. It
// grows as fusion gains features: what does not fuse yet is checked as
// unfused, and moves over once it does.
//
// Today the `mul` fuses into the sum's fold. The rest stays outside: sibling
// reductions, a reduction of a reduction, a full-row consumer and a view of a
// result.

// CHECK-LABEL: mo.graph @aroundReductions
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8, 128], f32>)
// CHECK:       %[[S:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  (%[[X]]: !mo.tensor<[8, 128], f32>)
// CHECK:       map.iter.reduce @builtin_kernels.monoids.ReduceSum
// CHECK:       map.call.elementwise @builtin_kernels.elementwise.Mul
// CHECK:       %[[M:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  (%[[X]]: !mo.tensor<[8, 128], f32>)
// CHECK:       map.iter.reduce @builtin_kernels.monoids.ReduceMax
// CHECK:       map.iter.elementwise<[8, 128]>
// CHECK-NEXT:  inputs  (%[[X]]: !mo.tensor<[8, 128], f32>, %[[S]]: !mo.tensor<[8, 1], f32>)
// CHECK:       map.call.elementwise @builtin_kernels.elementwise.Div
// CHECK:       map.iter.elementwise<[8]>
// CHECK-NEXT:  inputs  (%[[M]]: !mo.tensor<[8, 1], f32>)
// CHECK:       map.iter.rowwise<[8, 1], axis = 0>
// CHECK-NEXT:  inputs  (%[[M]]: !mo.tensor<[8, 1], f32>)
// CHECK:       map.iter.reduce @builtin_kernels.monoids.ReduceMin
mo.graph @aroundReductions(%x: !mo.tensor<[8, 128], f32>) -> (!mo.tensor<[8, 128], f32>, !mo.tensor<[8], f32>, !mo.tensor<[1, 1], f32>, !mo.tensor<[8, 1], f32>) {
  %sq = mo.mul(%x, %x) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%sq) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %m = mo.reduce.max(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %d = mo.div(%x, %sb) : !mo.tensor<[8, 128], f32>
  %mr = mo.static.reshape(%m) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8], f32>
  %mm = mo.reduce.min(%m) : (!mo.tensor<[8, 1], f32>) -> !mo.tensor<[1, 1], f32>
  mo.output %d, %mr, %mm, %s : !mo.tensor<[8, 128], f32>, !mo.tensor<[8], f32>, !mo.tensor<[1, 1], f32>, !mo.tensor<[8, 1], f32>
}
