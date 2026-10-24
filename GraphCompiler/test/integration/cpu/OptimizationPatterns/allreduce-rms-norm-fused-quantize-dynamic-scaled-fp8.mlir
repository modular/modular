// COM: Test AllReduce + RMSNorm + FP8 dynamic quantization op fusion
// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s

mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8<total_seq_len>(
  %input0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %input1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %input2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %input3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %sign_buff0: !mo.buffer<[537919488], ui8, gpu:0>,
  %sign_buff1: !mo.buffer<[537919488], ui8, gpu:1>,
  %sign_buff2: !mo.buffer<[537919488], ui8, gpu:2>,
  %sign_buff3: !mo.buffer<[537919488], ui8, gpu:3>,
  %c: !mo.chain,
  %residual0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %residual1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %residual2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %residual3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %gamma0: !mo.tensor<[16384], bf16, gpu:0>,
  %gamma1: !mo.tensor<[16384], bf16, gpu:1>,
  %gamma2: !mo.tensor<[16384], bf16, gpu:2>,
  %gamma3: !mo.tensor<[16384], bf16, gpu:3>
) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>,
      !mo.chain) {
    %allred:4, %out_chain = mo.distributed.allreduce.sum(%input0, %input1, %input2, %input3, %sign_buff0, %sign_buff1, %sign_buff2, %sign_buff3, %c) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.buffer<[537919488], ui8, gpu:2>, !mo.buffer<[537919488], ui8, gpu:3>, !mo.chain) -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.chain)

    %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>

    // Add + RMNSNorm + Quant (GPU:0)
    %add0 = mo.add(%residual0, %allred#0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm0 = mo.reduce.rms_norm(%add0, %gamma0) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[16384], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm_quant0:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 0>, parameters = {group_size_or_per_token = 16384 : index}](%norm0, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)

    // Add + RMNSNorm + Quant (GPU:1)
    %add1 = mo.add(%residual1, %allred#1) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm1 = mo.reduce.rms_norm(%add1, %gamma1) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[16384], bf16, gpu:1>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm_quant1:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 1>, parameters = {group_size_or_per_token = 16384 : index}](%norm1, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[1, total_seq_len], bf16, gpu:1>)

    // Add + RMNSNorm + Quant (GPU:2)
    %add2 = mo.add(%residual2, %allred#2) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>
    %norm2 = mo.reduce.rms_norm(%add2, %gamma2) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[16384], bf16, gpu:2>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>
    %norm_quant2:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 2>, parameters = {group_size_or_per_token = 16384 : index}](%norm2, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>, !mo.tensor<[1, total_seq_len], bf16, gpu:2>)

    // Add + RMNSNorm + Quant (GPU:3)
    %add3 = mo.add(%residual3, %allred#3) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
    %norm3 = mo.reduce.rms_norm(%add3, %gamma3) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.tensor<[16384], bf16, gpu:3>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
    %norm_quant3:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 3>, parameters = {group_size_or_per_token = 16384 : index}](%norm3, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>, !mo.tensor<[1, total_seq_len], bf16, gpu:3>)

    mo.output %norm_quant0#0, %norm_quant1#0, %norm_quant2#0, %norm_quant3#0, %out_chain : !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>, !mo.chain
}

// CHECK-LABEL: mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8
// CHECK: mo.composite.distributed.allreduce_add_rms_norm_quant_fp8(
// CHECK-SAME: {epsilon = #M.dense_array<9.99999997E-7> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>}

mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_residual_use<total_seq_len>(
  %input0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %input1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %input2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %input3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %sign_buff0: !mo.buffer<[537919488], ui8, gpu:0>,
  %sign_buff1: !mo.buffer<[537919488], ui8, gpu:1>,
  %sign_buff2: !mo.buffer<[537919488], ui8, gpu:2>,
  %sign_buff3: !mo.buffer<[537919488], ui8, gpu:3>,
  %c: !mo.chain,
  %residual0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %residual1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %residual2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %residual3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %gamma0: !mo.tensor<[16384], bf16, gpu:0>,
  %gamma1: !mo.tensor<[16384], bf16, gpu:1>,
  %gamma2: !mo.tensor<[16384], bf16, gpu:2>,
  %gamma3: !mo.tensor<[16384], bf16, gpu:3>,
  %other0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %other1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %other2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %other3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>,
      !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
      !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
      !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
      !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
      !mo.chain) {
    %allred:4, %out_chain = mo.distributed.allreduce.sum(%input0, %input1, %input2, %input3, %sign_buff0, %sign_buff1, %sign_buff2, %sign_buff3, %c) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.buffer<[537919488], ui8, gpu:2>, !mo.buffer<[537919488], ui8, gpu:3>, !mo.chain) -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.chain)

    %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>

    // Add + RMNSNorm + Quant + Residual use (GPU:0)
    %add0 = mo.add(%residual0, %allred#0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm0 = mo.reduce.rms_norm(%add0, %gamma0) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[16384], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm_quant0:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 0>, parameters = {group_size_or_per_token = 16384 : index}](%norm0, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)
    %out0 = mo.add(%other0, %add0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>

    // Add + RMNSNorm + Quant + Residual use (GPU:1)
    %add1 = mo.add(%residual1, %allred#1) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm1 = mo.reduce.rms_norm(%add1, %gamma1) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[16384], bf16, gpu:1>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm_quant1:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 1>, parameters = {group_size_or_per_token = 16384 : index}](%norm1, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[1, total_seq_len], bf16, gpu:1>)
    %out1 = mo.add(%other1, %add1) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>

    // Add + RMNSNorm + Quant + Residual use (GPU:2)
    %add2 = mo.add(%residual2, %allred#2) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>
    %norm2 = mo.reduce.rms_norm(%add2, %gamma2) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[16384], bf16, gpu:2>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>
    %norm_quant2:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 2>, parameters = {group_size_or_per_token = 16384 : index}](%norm2, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>, !mo.tensor<[1, total_seq_len], bf16, gpu:2>)
    %out2 = mo.add(%other2, %add2) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>

    // Add + RMNSNorm + Quant + Residual use (GPU:3)
    %add3 = mo.add(%residual3, %allred#3) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
    %norm3 = mo.reduce.rms_norm(%add3, %gamma3) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.tensor<[16384], bf16, gpu:3>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
    %norm_quant3:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 3>, parameters = {group_size_or_per_token = 16384 : index}](%norm3, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>, !mo.tensor<[1, total_seq_len], bf16, gpu:3>)
    %out3 = mo.add(%other3, %add3) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>

    mo.output %norm_quant0#0, %norm_quant1#0, %norm_quant2#0, %norm_quant3#0, %out0, %out1, %out2, %out3, %out_chain : !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.chain
}
// CHECK-LABEL: mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_residual_use
// CHECK: %[[OUTS_V:.*]]:4, %[[OUTS_S:.*]]:4, %[[OUTS_R:.*]]:4, %[[OUT0_C:.*]] = mo.composite.distributed.allreduce_add_rms_norm_quant_fp8(
// CHECK: %[[OUT0:.*]] = mo.add(%{{.*}}, %[[OUTS_R]]#0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
// CHECK: %[[OUT1:.*]] = mo.add(%{{.*}}, %[[OUTS_R]]#1) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
// CHECK: %[[OUT2:.*]] = mo.add(%{{.*}}, %[[OUTS_R]]#2) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>
// CHECK: %[[OUT3:.*]] = mo.add(%{{.*}}, %[[OUTS_R]]#3) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
// CHECK: mo.output %[[OUTS_V]]#0, %[[OUTS_V]]#1, %[[OUTS_V]]#2, %[[OUTS_V]]#3, %[[OUT0]], %[[OUT1]], %[[OUT2]], %[[OUT3]]

mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_amax_floor<total_seq_len>(
  %input0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %input1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %input2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %input3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %sign_buff0: !mo.buffer<[537919488], ui8, gpu:0>,
  %sign_buff1: !mo.buffer<[537919488], ui8, gpu:1>,
  %sign_buff2: !mo.buffer<[537919488], ui8, gpu:2>,
  %sign_buff3: !mo.buffer<[537919488], ui8, gpu:3>,
  %c: !mo.chain,
  %residual0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %residual1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %residual2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %residual3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %gamma0: !mo.tensor<[16384], bf16, gpu:0>,
  %gamma1: !mo.tensor<[16384], bf16, gpu:1>,
  %gamma2: !mo.tensor<[16384], bf16, gpu:2>,
  %gamma3: !mo.tensor<[16384], bf16, gpu:3>
) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>,
      !mo.chain) {
    %allred:4, %out_chain = mo.distributed.allreduce.sum(%input0, %input1, %input2, %input3, %sign_buff0, %sign_buff1, %sign_buff2, %sign_buff3, %c) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.buffer<[537919488], ui8, gpu:2>, !mo.buffer<[537919488], ui8, gpu:3>, !mo.chain) -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.chain)

    %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>

    // Add + RMNSNorm + Quant (GPU:0)
    %add0 = mo.add(%residual0, %allred#0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm0 = mo.reduce.rms_norm(%add0, %gamma0) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[16384], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm_quant0:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 0>, parameters = {amax_floor = "0.0001", group_size_or_per_token = 16384 : index}](%norm0, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)

    // Add + RMNSNorm + Quant (GPU:1)
    %add1 = mo.add(%residual1, %allred#1) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm1 = mo.reduce.rms_norm(%add1, %gamma1) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[16384], bf16, gpu:1>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm_quant1:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 1>, parameters = {amax_floor = "0.0001", group_size_or_per_token = 16384 : index}](%norm1, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[1, total_seq_len], bf16, gpu:1>)

    // Add + RMNSNorm + Quant (GPU:2)
    %add2 = mo.add(%residual2, %allred#2) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>
    %norm2 = mo.reduce.rms_norm(%add2, %gamma2) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[16384], bf16, gpu:2>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>
    %norm_quant2:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 2>, parameters = {amax_floor = "0.0001", group_size_or_per_token = 16384 : index}](%norm2, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>, !mo.tensor<[1, total_seq_len], bf16, gpu:2>)

    // Add + RMNSNorm + Quant (GPU:3)
    %add3 = mo.add(%residual3, %allred#3) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
    %norm3 = mo.reduce.rms_norm(%add3, %gamma3) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.tensor<[16384], bf16, gpu:3>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
    %norm_quant3:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 3>, parameters = {amax_floor = "0.0001", group_size_or_per_token = 16384 : index}](%norm3, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>, !mo.tensor<[1, total_seq_len], bf16, gpu:3>)

    mo.output %norm_quant0#0, %norm_quant1#0, %norm_quant2#0, %norm_quant3#0, %out_chain : !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>, !mo.chain
}

// A quantize op carrying an `amax_floor` blocks the fusion: the fused kernel
// has no floor.
// CHECK-LABEL: mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_amax_floor
// CHECK-NOT: mo.composite.distributed.allreduce_add_rms_norm_quant_fp8
// CHECK: mo.distributed.allreduce.sum
// CHECK-NOT: mo.composite.distributed.allreduce_add_rms_norm_quant_fp8
// CHECK-COUNT-4: symbol = "mo.quantize_dynamic_scaled_float8"{{.*}}amax_floor = "0.0001"
