// RUN: mo-opt %s | mo-opt | FileCheck %s

// `mo.sequence` round-trips: a stream-tagged region with a plain variadic
// input list mapped 1:1 to block arguments and `mo.yield` mapped to results.
// The custom region format prints block-argument names without types (they
// mirror the operands) and omits the explicit entry-block header.

// CHECK-LABEL: mo.graph @sequence_roundtrip
// CHECK: mo.sequence[1] (%{{.*}} : !mo.tensor<[2, 2], f32, gpu:0>) (%[[ARG:.*]]) -> (!mo.tensor<[2, 2], f32, gpu:0>) {
// CHECK:   %[[R:.*]] = mo.relu(%[[ARG]])
// CHECK:   mo.yield %[[R]]
mo.graph @sequence_roundtrip(%a: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %s = mo.sequence[1] (%a : !mo.tensor<[2, 2], f32, gpu:0>) (%arg) -> (!mo.tensor<[2, 2], f32, gpu:0>) {
    %r = mo.relu(%arg) : !mo.tensor<[2, 2], f32, gpu:0>
    mo.yield %r : !mo.tensor<[2, 2], f32, gpu:0>
  }
  mo.output %s : !mo.tensor<[2, 2], f32, gpu:0>
}

// A default-stream (streamId 0) sequence with multiple inputs/results also
// round-trips.

// CHECK-LABEL: mo.graph @sequence_default_stream
// CHECK: mo.sequence[0] (%{{.*}}, %{{.*}} : !mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:0>) (%{{.*}}, %{{.*}}) ->
mo.graph @sequence_default_stream(%a: !mo.tensor<[2], f32, gpu:0>, %b: !mo.tensor<[2], f32, gpu:0>) -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:0>) {
  %r:2 = mo.sequence[0] (%a, %b : !mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:0>) (%x, %y) -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:0>) {
    %rx = mo.relu(%x) : !mo.tensor<[2], f32, gpu:0>
    %ry = mo.relu(%y) : !mo.tensor<[2], f32, gpu:0>
    mo.yield %rx, %ry : !mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:0>
  }
  mo.output %r#0, %r#1 : !mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:0>
}
