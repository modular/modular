// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-elementwise | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-elementwise=large-tensor-element-threshold=16384 | FileCheck %s --check-prefix=HIGHTHRESH

// A multi-use binary op over TWO large operands must NOT be duplicated: the Add
// is materialized into its own kernel and each consumer reads its result.
mo.graph @binaryAddLargeFanout(%arg0: !mo.tensor<[4096, 2], f32, gpu:0>, %arg1: !mo.tensor<[4096, 2], f32, gpu:0>) -> (!mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>) {
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<[4096, 2], f32, gpu:0>
  %1 = mo.tanh(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  %2 = mo.relu(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  mo.output %1, %2 : !mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @binaryAddLargeFanout
// CHECK: %[[ADD:.*]] = map.iter.elementwise
// CHECK: map.call.elementwise @{{.*}}.Add
// CHECK-NOT: map.call.elementwise @{{.*}}.Add
// CHECK: map.iter.elementwise
// CHECK: inputs (%[[ADD]]
// CHECK: map.call.elementwise @{{.*}}.Tanh
// CHECK: map.iter.elementwise
// CHECK: inputs (%[[ADD]]
// CHECK: map.call.elementwise @{{.*}}.ReLU

// With `large-tensor-element-threshold` raised above the operands' 8192-element
// count, both operands count as small, so the Add is cloned into each
// consumer's fused kernel instead of being materialized once.
// HIGHTHRESH-LABEL: mo.graph @binaryAddLargeFanout
// HIGHTHRESH: map.iter.elementwise
// HIGHTHRESH: map.call.elementwise @{{.*}}.Add
// HIGHTHRESH: map.call.elementwise @{{.*}}.Tanh
// HIGHTHRESH: map.iter.elementwise
// HIGHTHRESH: map.call.elementwise @{{.*}}.Add
// HIGHTHRESH: map.call.elementwise @{{.*}}.ReLU

// A multi-use UNARY op has at most one large operand, so re-reading it on
// duplication is not a net loss. It is still cloned into each consumer's fused
// subgraph (Exp appears in both kernels), preserving the existing fusion win.
mo.graph @unaryExpLargeFanout(%arg0: !mo.tensor<[4096, 2], f32, gpu:0>) -> (!mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>) {
  %0 = mo.exp(%arg0) : !mo.tensor<[4096, 2], f32, gpu:0>
  %1 = mo.tanh(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  %2 = mo.relu(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  mo.output %1, %2 : !mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @unaryExpLargeFanout
// CHECK: map.iter.elementwise
// CHECK: map.call.elementwise @{{.*}}.Exp
// CHECK: map.call.elementwise @{{.*}}.Tanh
// CHECK: map.iter.elementwise
// CHECK: map.call.elementwise @{{.*}}.Exp
// CHECK: map.call.elementwise @{{.*}}.ReLU

// The same unary op, but its result is also a graph output: it is written to
// memory either way, so cloning it into the consumer only adds work.
mo.graph @unaryExpEscapingOutput(%arg0: !mo.tensor<[4096, 2], f32, gpu:0>) -> (!mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>) {
  %0 = mo.exp(%arg0) : !mo.tensor<[4096, 2], f32, gpu:0>
  %1 = mo.tanh(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  mo.output %0, %1 : !mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @unaryExpEscapingOutput
// CHECK: map.iter.elementwise
// CHECK: map.call.elementwise @{{.*}}.Exp
// CHECK: map.iter.elementwise
// CHECK-NOT: map.call.elementwise @{{.*}}.Exp
// CHECK: map.call.elementwise @{{.*}}.Tanh

// The same unary op feeding an opaque kernel that cannot absorb it: the matmul
// reads the materialized result, so cloning into the other consumer only adds
// work.
mo.graph @unaryExpIntoOpaque(%arg0: !mo.tensor<[4096, 2], f32, gpu:0>, %w: !mo.tensor<[2, 8], f32, gpu:0>) -> (!mo.tensor<[4096, 8], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>) {
  %0 = mo.exp(%arg0) : !mo.tensor<[4096, 2], f32, gpu:0>
  %1 = mo.matmul(%0, %w) : (!mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[2, 8], f32, gpu:0>) -> !mo.tensor<[4096, 8], f32, gpu:0>
  %2 = mo.tanh(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  mo.output %1, %2 : !mo.tensor<[4096, 8], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @unaryExpIntoOpaque
// CHECK: map.iter.elementwise
// CHECK: map.call.elementwise @{{.*}}.Exp
// CHECK: map.iter.opaque
// CHECK: map.call.opaque @{{.*}}.Matmul
// CHECK: map.iter.elementwise
// CHECK-NOT: map.call.elementwise @{{.*}}.Exp
// CHECK: map.call.elementwise @{{.*}}.Tanh
