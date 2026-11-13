// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// A concat is a loop reading its inputs through `map.index.concat.select`.
// A reduction doesn't fuse it yet: the concat stays its own kernel, and the
// reduction reads what it stores.

// CHECK-LABEL: mo.graph @sum_of_concat
// CHECK:       %[[C:.*]] = map.iter.elementwise<[4, 16]>
// CHECK:       map.index.concat.select<#map.index.concat.spec<[4, 6], [4, 10] -> [4, 16], axis = 1>>
// CHECK:       map.iter.rowwise<[4, 16], axis = 1>
// CHECK-NEXT:  inputs (%[[C]]
// CHECK-NOT:   map.index.concat.select
// CHECK:       mo.output
mo.graph @sum_of_concat(%a: !mo.tensor<[4, 6], f32>, %b: !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 1], f32> {
  %c = mo.concat[1] (%a, %b) : (!mo.tensor<[4, 6], f32>, !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32>
  %r = mo.reduce.add(%c) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  mo.output %r : !mo.tensor<[4, 1], f32>
}
