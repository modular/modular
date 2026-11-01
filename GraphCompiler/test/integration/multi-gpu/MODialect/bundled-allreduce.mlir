// RUN: mo-opt %s --canonicalize | FileCheck %s
// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o /dev/null

// Test that mo.bundled.allreduce.sum inside mo.parallel round-trips through
// canonicalization and compiles through the full mo-to-mogg pipeline.
// NOTE: This test requires a system with at least 2 GPUs (JIT compilation
// of the allreduce kernel needs a known GPU architecture).

// CHECK-LABEL: mo.graph @bundled_allreduce
mo.graph @bundled_allreduce(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %in0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>> : !mo.tensor<[3], f32, gpu:0>
  %in1 = mo.constant <#M.dense_array<4.0, 5.0, 6.0> : tensor<3xf32>> : !mo.tensor<[3], f32, gpu:1>

  %dt = mo.tensor.bundle(%in0, %in1) : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>) -> (!mo.bundle<[3], f32, gpu:0-1>)
  %ch = mo.chain.create()
  // CHECK: mo.parallel
  // CHECK: mo.bundled.expand
  // CHECK: mo.bundled.allreduce.sum
  // The in-chain enters the body through the trailing chain block arg %bch;
  // the collective consumes it and the body yields the resulting out-chain.
  %res_bundle, %out_ch = mo.parallel (%arg, %bch) in (%dt : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>, %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch) -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %peer0, %peer1 = mo.bundled.expand(%arg) : !mo.tensor<[3], f32, gpu:0> -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
    %out, %ch_out = mo.bundled.allreduce.sum(%peer0, %peer1, %sig0, %sig1, %bch)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }

  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[3], f32, gpu:0-1>) -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  %res0_host, %ch3 = mo.transfer[%out_ch] %res_0 : !mo.tensor<[3], f32, gpu:0> to <"cpu", 0>
  %res1_host, %ch4 = mo.transfer[%ch3] %res_1 : !mo.tensor<[3], f32, gpu:1> to <"cpu", 0>
  mo.output %res0_host, %res1_host : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// The body-local-chain regression test (mo.chain.create inside mo.parallel)
// is restored alongside per-launch lowering of view/chain ops. See GEX-3701.
