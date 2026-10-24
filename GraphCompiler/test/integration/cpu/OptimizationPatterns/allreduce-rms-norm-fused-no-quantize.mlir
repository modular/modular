// COM: An allreduce + add + rms_norm chain with no trailing FP8 quantize
// COM: fuses into the same composite, in the input dtype.
// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s

mo.graph @all_reduce_add_rms_norm_bf16<seq>(
  %in0: !mo.tensor<[seq, 8], bf16, gpu:0>,
  %in1: !mo.tensor<[seq, 8], bf16, gpu:1>,
  %sig0: !mo.buffer<[4096], ui8, gpu:0>,
  %sig1: !mo.buffer<[4096], ui8, gpu:1>,
  %c: !mo.chain,
  %res0: !mo.tensor<[seq, 8], bf16, gpu:0>,
  %res1: !mo.tensor<[seq, 8], bf16, gpu:1>,
  %gamma0: !mo.tensor<[8], bf16, gpu:0>,
  %gamma1: !mo.tensor<[8], bf16, gpu:1>
) -> (!mo.tensor<[seq, 8], bf16, gpu:0>, !mo.tensor<[seq, 8], bf16, gpu:1>, !mo.chain) {
  %allred:2, %out_chain = mo.distributed.allreduce.sum(%in0, %in1, %sig0, %sig1, %c) : (!mo.tensor<[seq, 8], bf16, gpu:0>, !mo.tensor<[seq, 8], bf16, gpu:1>, !mo.buffer<[4096], ui8, gpu:0>, !mo.buffer<[4096], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[seq, 8], bf16, gpu:0>, !mo.tensor<[seq, 8], bf16, gpu:1>, !mo.chain)

  %add0 = mo.add(%res0, %allred#0) : !mo.tensor<[seq, 8], bf16, gpu:0>
  %norm0 = mo.reduce.rms_norm(%add0, %gamma0) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[seq, 8], bf16, gpu:0>, !mo.tensor<[8], bf16, gpu:0>) -> !mo.tensor<[seq, 8], bf16, gpu:0>

  %add1 = mo.add(%res1, %allred#1) : !mo.tensor<[seq, 8], bf16, gpu:1>
  %norm1 = mo.reduce.rms_norm(%add1, %gamma1) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[seq, 8], bf16, gpu:1>, !mo.tensor<[8], bf16, gpu:1>) -> !mo.tensor<[seq, 8], bf16, gpu:1>

  mo.output %norm0, %norm1, %out_chain : !mo.tensor<[seq, 8], bf16, gpu:0>, !mo.tensor<[seq, 8], bf16, gpu:1>, !mo.chain
}

// The scale the composite still carries replaces the inner dim.
// CHECK-LABEL: mo.graph @all_reduce_add_rms_norm_bf16
// CHECK: mo.composite.distributed.allreduce_add_rms_norm_quant_fp8(
// CHECK-SAME: {epsilon = #M.dense_array<9.99999997E-7> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>}
// CHECK-SAME: !mo.tensor<[seq, 1], f32, gpu:0>

// The fused kernel needs P2P between the peers, so a CPU chain must not fuse.
mo.graph @all_reduce_add_rms_norm_cpu<seq>(
  %in0: !mo.tensor<[seq, 8], f32, cpu:0>,
  %in1: !mo.tensor<[seq, 8], f32, cpu:1>,
  %sig0: !mo.buffer<[4096], ui8, cpu:0>,
  %sig1: !mo.buffer<[4096], ui8, cpu:1>,
  %c: !mo.chain,
  %res0: !mo.tensor<[seq, 8], f32, cpu:0>,
  %res1: !mo.tensor<[seq, 8], f32, cpu:1>,
  %gamma0: !mo.tensor<[8], f32, cpu:0>,
  %gamma1: !mo.tensor<[8], f32, cpu:1>
) -> (!mo.tensor<[seq, 8], f32, cpu:0>, !mo.tensor<[seq, 8], f32, cpu:1>, !mo.chain) {
  %allred:2, %out_chain = mo.distributed.allreduce.sum(%in0, %in1, %sig0, %sig1, %c) : (!mo.tensor<[seq, 8], f32, cpu:0>, !mo.tensor<[seq, 8], f32, cpu:1>, !mo.buffer<[4096], ui8, cpu:0>, !mo.buffer<[4096], ui8, cpu:1>, !mo.chain) -> (!mo.tensor<[seq, 8], f32, cpu:0>, !mo.tensor<[seq, 8], f32, cpu:1>, !mo.chain)

  %add0 = mo.add(%res0, %allred#0) : !mo.tensor<[seq, 8], f32, cpu:0>
  %norm0 = mo.reduce.rms_norm(%add0, %gamma0) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[seq, 8], f32, cpu:0>, !mo.tensor<[8], f32, cpu:0>) -> !mo.tensor<[seq, 8], f32, cpu:0>

  %add1 = mo.add(%res1, %allred#1) : !mo.tensor<[seq, 8], f32, cpu:1>
  %norm1 = mo.reduce.rms_norm(%add1, %gamma1) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[seq, 8], f32, cpu:1>, !mo.tensor<[8], f32, cpu:1>) -> !mo.tensor<[seq, 8], f32, cpu:1>

  mo.output %norm0, %norm1, %out_chain : !mo.tensor<[seq, 8], f32, cpu:0>, !mo.tensor<[seq, 8], f32, cpu:1>, !mo.chain
}

// CHECK-LABEL: mo.graph @all_reduce_add_rms_norm_cpu
// CHECK: mo.distributed.allreduce.sum
// CHECK-NOT: mo.composite.distributed.allreduce_add_rms_norm_quant_fp8

// The composite carries one epsilon for every device, so per-device norms with
// different epsilons must not fuse.
mo.graph @all_reduce_add_rms_norm_mismatched_epsilon<seq>(
  %in0: !mo.tensor<[seq, 8], bf16, gpu:0>,
  %in1: !mo.tensor<[seq, 8], bf16, gpu:1>,
  %sig0: !mo.buffer<[4096], ui8, gpu:0>,
  %sig1: !mo.buffer<[4096], ui8, gpu:1>,
  %c: !mo.chain,
  %res0: !mo.tensor<[seq, 8], bf16, gpu:0>,
  %res1: !mo.tensor<[seq, 8], bf16, gpu:1>,
  %gamma0: !mo.tensor<[8], bf16, gpu:0>,
  %gamma1: !mo.tensor<[8], bf16, gpu:1>
) -> (!mo.tensor<[seq, 8], bf16, gpu:0>, !mo.tensor<[seq, 8], bf16, gpu:1>, !mo.chain) {
  %allred:2, %out_chain = mo.distributed.allreduce.sum(%in0, %in1, %sig0, %sig1, %c) : (!mo.tensor<[seq, 8], bf16, gpu:0>, !mo.tensor<[seq, 8], bf16, gpu:1>, !mo.buffer<[4096], ui8, gpu:0>, !mo.buffer<[4096], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[seq, 8], bf16, gpu:0>, !mo.tensor<[seq, 8], bf16, gpu:1>, !mo.chain)

  %add0 = mo.add(%res0, %allred#0) : !mo.tensor<[seq, 8], bf16, gpu:0>
  %norm0 = mo.reduce.rms_norm(%add0, %gamma0) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[seq, 8], bf16, gpu:0>, !mo.tensor<[8], bf16, gpu:0>) -> !mo.tensor<[seq, 8], bf16, gpu:0>

  %add1 = mo.add(%res1, %allred#1) : !mo.tensor<[seq, 8], bf16, gpu:1>
  %norm1 = mo.reduce.rms_norm(%add1, %gamma1) {epsilon = #M.dense_array<1.0e-05> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[seq, 8], bf16, gpu:1>, !mo.tensor<[8], bf16, gpu:1>) -> !mo.tensor<[seq, 8], bf16, gpu:1>

  mo.output %norm0, %norm1, %out_chain : !mo.tensor<[seq, 8], bf16, gpu:0>, !mo.tensor<[seq, 8], bf16, gpu:1>, !mo.chain
}

// CHECK-LABEL: mo.graph @all_reduce_add_rms_norm_mismatched_epsilon
// CHECK: mo.distributed.allreduce.sum
// CHECK-NOT: mo.composite.distributed.allreduce_add_rms_norm_quant_fp8
