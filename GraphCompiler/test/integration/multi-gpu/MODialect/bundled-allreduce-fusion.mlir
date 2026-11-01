// RUN: mo-opt %s --canonicalize | FileCheck %s
// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o /dev/null

// Verify per-kernel stub structure at the MGP stage.
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-pipeline --map-to-mogg --mogg-pipeline --mogg-to-mgp | FileCheck %s --check-prefix=CHECK-MGP

// Verify that MergeIdenticalFuncs deduplicates the per-launch stubs
// into a single shared stub (canonical arg ordering makes them identical).
// SymbolDCE removes the dead stubs whose callers were rewritten.
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-pipeline --map-to-mogg --mogg-pipeline --mogg-to-mgp \
// RUN:   --mogg-merge-identical-funcs --symbol-dce -o /dev/null

// Test that mo.bundled.allreduce.sum followed by mo.relu inside mo.parallel
// compiles through the full mo-to-mogg pipeline. This exercises both the
// multi-kernel and bundled-allreduce features together:
//   - Multiple kernels in a single parallel region (allreduce + relu)
//   - Bundled allreduce N-expanded so each launch sees all GPUs' data
//   - Intermediate value flow from allreduce output to relu input
//   - Separate compiled regions for allreduce and relu kernel groups

// NOTE: Requires a system with at least 2 GPUs.

// CHECK-LABEL: mo.graph @bundled_allreduce_then_relu
mo.graph @bundled_allreduce_then_relu(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %in0 = mo.constant <#M.dense_array<-1.0, 2.0, -3.0> : tensor<3xf32>> : !mo.tensor<[3], f32, gpu:0>
  %in1 = mo.constant <#M.dense_array<4.0, -5.0, 6.0> : tensor<3xf32>> : !mo.tensor<[3], f32, gpu:1>

  %dt = mo.tensor.bundle(%in0, %in1) : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>) -> (!mo.bundle<[3], f32, gpu:0-1>)
  %ch = mo.chain.create()
  // CHECK: mo.parallel
  // CHECK: mo.bundled.expand
  // CHECK: mo.bundled.allreduce.sum
  // CHECK: mo.relu
  %res_bundle, %out_ch = mo.parallel (%arg, %bch) in (%dt : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>, %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch) -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %peer0, %peer1 = mo.bundled.expand(%arg) : !mo.tensor<[3], f32, gpu:0> -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
    %out, %ch_out = mo.bundled.allreduce.sum(%peer0, %peer1, %sig0, %sig1, %bch)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    %relu = mo.relu(%out) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %relu, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }

  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[3], f32, gpu:0-1>) -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  %res0_host, %ch3 = mo.transfer[%out_ch] %res_0 : !mo.tensor<[3], f32, gpu:0> to <"cpu", 0>
  %res1_host, %ch4 = mo.transfer[%ch3] %res_1 : !mo.tensor<[3], f32, gpu:1> to <"cpu", 0>
  mo.output %res0_host, %res1_host : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// The fusion path folds relu into the bundled-allreduce epilogue, so a single
// fused stub is dispatched async to each GPU (one call per device) inside one
// compiled region, then joined over the two task handles.
// CHECK-MGP: mogg.routine.async
// CHECK-MGP:   mogg.call.async @{{__bundled_allreduce_sum[A-Za-z0-9_]*}}
// CHECK-MGP:   mogg.call.async @{{__bundled_allreduce_sum[A-Za-z0-9_]*}}
// CHECK-MGP:   mogg.async.join({{.*}})
