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

// logsoftmax as ops.logsoftmax spells it: the shifted row feeds both the sum
// and the final subtract, and the log runs once per row.
// CHECK-LABEL: mo.graph @logsoftmax
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  (%arg0: !mo.tensor<[8, 128], bf16>)
// CHECK:         map.iter.reduce @builtin_kernels.monoids.ReduceMax
// CHECK:         map.iter.reduce @builtin_kernels.monoids.ReduceSum
// CHECK:         map.call.elementwise @builtin_kernels.elementwise.Log
// CHECK:         map.iter.map
// CHECK-NOT:   map.iter.
// CHECK:       mo.output
mo.graph @logsoftmax(%x: !mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], bf16> {
  %xa = mo.cast(%x) : (!mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32>
  %m = mo.reduce.max(%xa) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %mb = mo.static.broadcast_to(%m) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %d = mo.sub(%xa, %mb) : !mo.tensor<[8, 128], f32>
  %e = mo.exp(%d) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%e) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %l = mo.log(%s) : !mo.tensor<[8, 1], f32>
  %lb = mo.static.broadcast_to(%l) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %y = mo.sub(%d, %lb) : !mo.tensor<[8, 128], f32>
  %yb = mo.cast(%y) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], bf16>
  mo.output %yb : !mo.tensor<[8, 128], bf16>
}
