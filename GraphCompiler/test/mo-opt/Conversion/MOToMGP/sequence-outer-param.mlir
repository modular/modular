// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp | FileCheck %s

// A parameter declared inside an `mo.sequence` body from a parameter declared
// outside it. The body sort analyzes the sequence alone, where the outer
// parameter has no declaring op, and must not take that as a provider.

// CHECK-LABEL: mgp.model @seq_outer_param
// CHECK: mgp.device_context.select_stream[1]
mo.graph @seq_outer_param<N>(%a: !mo.tensor<[N], f32, gpu:0>) -> !mo.tensor<[N], f32, gpu:0> {
  %s = mo.sequence[1] (%a : !mo.tensor<[N], f32, gpu:0>) (%arg) -> (!mo.tensor<[N], f32, gpu:0>) {
    %r = mo.relu(%arg) : !mo.tensor<[N], f32, gpu:0>
    kgen.param.declare Q: scalar<si64> = <N>
    %q = mo.rebind(%r) : !mo.tensor<[N], f32, gpu:0> -> !mo.tensor<[Q], f32, gpu:0>
    %r2 = mo.relu(%q) : !mo.tensor<[Q], f32, gpu:0>
    %n = mo.rebind(%r2) : !mo.tensor<[Q], f32, gpu:0> -> !mo.tensor<[N], f32, gpu:0>
    mo.yield %n : !mo.tensor<[N], f32, gpu:0>
  }
  mo.output %s : !mo.tensor<[N], f32, gpu:0>
}
