// RUN: mo-opt %s --mogg-import-packages --mo-tmp-decompose-composites --mo-to-map --map-fuse-kernels | FileCheck %s

// Softmax fuses into one rowwise kernel: the row of the input, a max fold, a
// sum fold that reads it, and a map phase.

// The decomposed composite.
// CHECK-LABEL: mo.graph @softmax
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  (%arg0: !mo.tensor<[8, 128], bf16>{{.*}})
// CHECK:         map.iter.reduce @builtin_kernels.monoids.ReduceMax
// CHECK:         map.iter.reduce @builtin_kernels.monoids.ReduceSum
// CHECK:         map.iter.map
// CHECK-NOT:   map.iter.
// CHECK:       mo.output
mo.graph @softmax(%x: !mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], bf16> {
  %y = mo.composite.reduce.softmax(%x) [axis = 1] : (!mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], bf16>
  mo.output %y : !mo.tensor<[8, 128], bf16>
}
