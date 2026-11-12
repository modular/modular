// `group_size_or_per_token` is a defaulted attribute: the per-token scale (`-1`)
// is its default, so a per-token composite prints without it and only a real
// group appears in the IR.
// RUN: mo-opt %s | FileCheck %s

// No attribute: the default, one scale a row.
mo.graph @omitted(
  %in0: !mo.tensor<[4, 8], bf16, gpu:0>, %in1: !mo.tensor<[4, 8], bf16, gpu:1>,
  %sig0: !mo.buffer<[64], ui8, gpu:0>, %sig1: !mo.buffer<[64], ui8, gpu:1>,
  %res0: !mo.tensor<[4, 8], bf16, gpu:0>, %res1: !mo.tensor<[4, 8], bf16, gpu:1>,
  %gamma0: !mo.tensor<[8], bf16, gpu:0>, %gamma1: !mo.tensor<[8], bf16, gpu:1>,
  %scale_ub: !mo.tensor<[], f32>,
  %c: !mo.chain
) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4, 8], f8e4m3fn, gpu:1>,
      !mo.chain) {
  %out:2, %scales:2, %res_out:2, %out_chain = mo.composite.distributed.allreduce_add_rms_norm_quant_fp8(%in0, %in1, %sig0, %sig1, %res0, %res1, %gamma0, %gamma1, %scale_ub, %scale_ub, %c) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.buffer<[64], ui8, gpu:0>, !mo.buffer<[64], ui8, gpu:1>, !mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.tensor<[8], bf16, gpu:0>, !mo.tensor<[8], bf16, gpu:1>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.chain) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4, 8], f8e4m3fn, gpu:1>, !mo.tensor<[1, 4], bf16, gpu:0>, !mo.tensor<[1, 4], bf16, gpu:1>, !mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.chain)
  mo.output %out#0, %out#1, %out_chain : !mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4, 8], f8e4m3fn, gpu:1>, !mo.chain
}
// CHECK-LABEL: mo.graph @omitted
// CHECK: mo.composite.distributed.allreduce_add_rms_norm_quant_fp8(
// CHECK-SAME: {epsilon = #M.dense_array<9.99999997E-7> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>, multiply_before_cast = true}

// Spelled out, the default elides as well.
mo.graph @explicit_per_token(
  %in0: !mo.tensor<[4, 8], bf16, gpu:0>, %in1: !mo.tensor<[4, 8], bf16, gpu:1>,
  %sig0: !mo.buffer<[64], ui8, gpu:0>, %sig1: !mo.buffer<[64], ui8, gpu:1>,
  %res0: !mo.tensor<[4, 8], bf16, gpu:0>, %res1: !mo.tensor<[4, 8], bf16, gpu:1>,
  %gamma0: !mo.tensor<[8], bf16, gpu:0>, %gamma1: !mo.tensor<[8], bf16, gpu:1>,
  %scale_ub: !mo.tensor<[], f32>,
  %c: !mo.chain
) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4, 8], f8e4m3fn, gpu:1>,
      !mo.chain) {
  %out:2, %scales:2, %res_out:2, %out_chain = mo.composite.distributed.allreduce_add_rms_norm_quant_fp8(%in0, %in1, %sig0, %sig1, %res0, %res1, %gamma0, %gamma1, %scale_ub, %scale_ub, %c) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true, group_size_or_per_token = -1} : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.buffer<[64], ui8, gpu:0>, !mo.buffer<[64], ui8, gpu:1>, !mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.tensor<[8], bf16, gpu:0>, !mo.tensor<[8], bf16, gpu:1>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.chain) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4, 8], f8e4m3fn, gpu:1>, !mo.tensor<[1, 4], bf16, gpu:0>, !mo.tensor<[1, 4], bf16, gpu:1>, !mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.chain)
  mo.output %out#0, %out#1, %out_chain : !mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4, 8], f8e4m3fn, gpu:1>, !mo.chain
}
// CHECK-LABEL: mo.graph @explicit_per_token
// CHECK: mo.composite.distributed.allreduce_add_rms_norm_quant_fp8(
// CHECK-SAME: {epsilon = #M.dense_array<9.99999997E-7> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>, multiply_before_cast = true}

// A group narrower than the row is kept, and scales a group at a time.
mo.graph @grouped(
  %in0: !mo.tensor<[4, 8], bf16, gpu:0>, %in1: !mo.tensor<[4, 8], bf16, gpu:1>,
  %sig0: !mo.buffer<[64], ui8, gpu:0>, %sig1: !mo.buffer<[64], ui8, gpu:1>,
  %res0: !mo.tensor<[4, 8], bf16, gpu:0>, %res1: !mo.tensor<[4, 8], bf16, gpu:1>,
  %gamma0: !mo.tensor<[8], bf16, gpu:0>, %gamma1: !mo.tensor<[8], bf16, gpu:1>,
  %scale_ub: !mo.tensor<[], f32>,
  %c: !mo.chain
) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4, 8], f8e4m3fn, gpu:1>,
      !mo.chain) {
  %out:2, %scales:2, %res_out:2, %out_chain = mo.composite.distributed.allreduce_add_rms_norm_quant_fp8(%in0, %in1, %sig0, %sig1, %res0, %res1, %gamma0, %gamma1, %scale_ub, %scale_ub, %c) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true, group_size_or_per_token = 4} : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.buffer<[64], ui8, gpu:0>, !mo.buffer<[64], ui8, gpu:1>, !mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.tensor<[8], bf16, gpu:0>, !mo.tensor<[8], bf16, gpu:1>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.chain) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4, 8], f8e4m3fn, gpu:1>, !mo.tensor<[2, 4], bf16, gpu:0>, !mo.tensor<[2, 4], bf16, gpu:1>, !mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.chain)
  mo.output %out#0, %out#1, %out_chain : !mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4, 8], f8e4m3fn, gpu:1>, !mo.chain
}
// CHECK-LABEL: mo.graph @grouped
// CHECK: mo.composite.distributed.allreduce_add_rms_norm_quant_fp8(
// CHECK-SAME: {epsilon = #M.dense_array<9.99999997E-7> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>, multiply_before_cast = true, group_size_or_per_token = 4}
