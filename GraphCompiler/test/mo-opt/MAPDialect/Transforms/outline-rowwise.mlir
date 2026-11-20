// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-outline-structured-kernel-bodies | FileCheck %s

// A `map.iter.rowwise` is wrapped in a `map.iter.opaque`, and its body moves
// into a `map.iter.rowwise.body` with the same shape and axis.

// CHECK-LABEL: mo.graph @sum
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8, 128], f32>)
// CHECK:       %[[R:.*]] = map.iter.opaque
// CHECK-NEXT:    inputs (%[[X]]: !mo.tensor<[8, 128], f32>)
// CHECK-NEXT:    outputs (%[[OUT:.*]] : !mo.tensor<[8, 1], f32>) {
// CHECK-NEXT:    map.iter.rowwise.body<[8, 128], axis = 1>
// CHECK-NEXT:      row (%[[ROW:.*]]: !map.index<2>)
// CHECK-NEXT:      capturing (%[[X0:.*]] = %[[X]] : !mo.tensor<[8, 128], f32>, %[[OUT0:.*]] = %[[OUT]] : !mo.tensor<[8, 1], f32>) {
// CHECK-NEXT:      %[[ACC:.*]] = map.iter.reduce @builtin_kernels.monoids.ReduceSum
// CHECK-NEXT:        map.load %[[X0]]
// CHECK:           map.store %[[OUT0]][%[[ROW]]] = %[[ACC]]
// CHECK:       } {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}}
// CHECK-NEXT:  mo.output %[[R]]
mo.graph @sum(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %r : !mo.tensor<[8, 1], f32>
}
