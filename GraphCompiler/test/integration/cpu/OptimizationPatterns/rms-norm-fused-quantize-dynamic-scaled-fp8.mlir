// COM: Test RMSNorm + FP8 dynamic quantization op fusion
// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s

mo.graph @rms_norm_quantize_dynamic_scaled_fp8<total_seq_len>(
  %input: !mo.tensor<[total_seq_len, 10], bf16, gpu:0>,
  %gamma: !mo.tensor<[10], bf16, gpu:0>
) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>) {
  %norm = mo.reduce.rms_norm(%input, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[10], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 10], bf16, gpu:0>
  %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>
  %norm_quant:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 0>, parameters = {group_size_or_per_token = 10 : index}](%norm, %scale_ub) : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)
  mo.output %norm_quant#0, %norm_quant#1 : !mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>
}
// CHECK-LABEL: mo.graph @rms_norm_quantize_dynamic_scaled_fp8<total_seq_len>
// CHECK: mo.composite.rms_norm_fused_quantize_dynamic_scaled_fp8
// CHECK-SAME: {epsilon = #M.dense_array<9.99999997E-7> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>}

mo.graph @rms_norm_quantize_dynamic_scaled_fp8_tokenwise<total_seq_len>(
  %input: !mo.tensor<[total_seq_len, 10], bf16, gpu:0>,
  %gamma: !mo.tensor<[10], bf16, gpu:0>
) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>) {
  %norm = mo.reduce.rms_norm(%input, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[10], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 10], bf16, gpu:0>
  %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>
  %norm_quant:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 0>, parameters = {group_size_or_per_token = -1 : index}](%norm, %scale_ub) : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)
  mo.output %norm_quant#0, %norm_quant#1 : !mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>
}
// CHECK-LABEL: mo.graph @rms_norm_quantize_dynamic_scaled_fp8_tokenwise<total_seq_len>
// CHECK: mo.composite.rms_norm_fused_quantize_dynamic_scaled_fp8

// multiply_before_cast = true is explicit, so this pattern matches.
mo.graph @rms_norm_quantize_dynamic_scaled_fp8_multiply_before_cast_missing<total_seq_len>(
  %input: !mo.tensor<[total_seq_len, 10], bf16, gpu:0>,
  %gamma: !mo.tensor<[10], bf16, gpu:0>
) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>) {
  %norm = mo.reduce.rms_norm(%input, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[10], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 10], bf16, gpu:0>
  %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>
  %norm_quant:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 0>, parameters = {group_size_or_per_token = 10 : index}](%norm, %scale_ub) : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)
  mo.output %norm_quant#0, %norm_quant#1 : !mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>
}
// CHECK-LABEL: mo.graph @rms_norm_quantize_dynamic_scaled_fp8_multiply_before_cast_missing<total_seq_len>
// CHECK: mo.composite.rms_norm_fused_quantize_dynamic_scaled_fp8

mo.graph @rms_norm_quantize_dynamic_scaled_fp8_device_wrong<total_seq_len>(
  %input: !mo.tensor<[total_seq_len, 10], bf16, cpu:0>,
  %gamma: !mo.tensor<[10], bf16, cpu:0>
) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, cpu:0>, !mo.tensor<[1, total_seq_len], bf16, cpu:0>) {
  %norm = mo.reduce.rms_norm(%input, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 10], bf16, cpu:0>, !mo.tensor<[10], bf16, cpu:0>) -> !mo.tensor<[total_seq_len, 10], bf16, cpu:0>
  %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>
  %norm_quant:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"cpu", 0>, parameters = {group_size_or_per_token = 10 : index}](%norm, %scale_ub) : (!mo.tensor<[total_seq_len, 10], bf16, cpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, cpu:0>, !mo.tensor<[1, total_seq_len], bf16, cpu:0>)
  mo.output %norm_quant#0, %norm_quant#1 : !mo.tensor<[total_seq_len, 10], f8e4m3fn, cpu:0>, !mo.tensor<[1, total_seq_len], bf16, cpu:0>
}
// CHECK-LABEL: mo.graph @rms_norm_quantize_dynamic_scaled_fp8_device_wrong<total_seq_len>
// CHECK: mo.reduce.rms_norm
// CHECK: "mo.quantize_dynamic_scaled_float8"

mo.graph @rms_norm_quantize_dynamic_scaled_fp8_multiply_before_cast_wrong<total_seq_len>(
  %input: !mo.tensor<[total_seq_len, 10], bf16, gpu:0>,
  %gamma: !mo.tensor<[10], bf16, gpu:0>
) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>) {
  %norm = mo.reduce.rms_norm(%input, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[10], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 10], bf16, gpu:0>
  %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>
  %norm_quant:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 0>, parameters = {group_size_or_per_token = 10 : index}](%norm, %scale_ub) : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)
  mo.output %norm_quant#0, %norm_quant#1 : !mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>
}
// CHECK-LABEL: mo.graph @rms_norm_quantize_dynamic_scaled_fp8_multiply_before_cast_wrong<total_seq_len>
// CHECK: mo.reduce.rms_norm
// CHECK: "mo.quantize_dynamic_scaled_float8"

mo.graph @rms_norm_quantize_dynamic_scaled_fp8_group_size_wrong<total_seq_len>(
  %input: !mo.tensor<[total_seq_len, 10], bf16, gpu:0>,
  %gamma: !mo.tensor<[10], bf16, gpu:0>
) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>) {
  %norm = mo.reduce.rms_norm(%input, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[10], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 10], bf16, gpu:0>
  %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>
  %norm_quant:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 0>, parameters = {group_size_or_per_token = 5 : index}](%norm, %scale_ub) : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)
  mo.output %norm_quant#0, %norm_quant#1 : !mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>
}
// CHECK-LABEL: mo.graph @rms_norm_quantize_dynamic_scaled_fp8_group_size_wrong<total_seq_len>
// CHECK: mo.reduce.rms_norm
// CHECK: "mo.quantize_dynamic_scaled_float8"

mo.graph @rms_norm_quantize_dynamic_scaled_fp8_group_size_missing<total_seq_len>(
  %input: !mo.tensor<[total_seq_len, 10], bf16, gpu:0>,
  %gamma: !mo.tensor<[10], bf16, gpu:0>
) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>) {
  %norm = mo.reduce.rms_norm(%input, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[10], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 10], bf16, gpu:0>
  %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>
  %norm_quant:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 0>, parameters = {unknown_parameter = 10 : index}](%norm, %scale_ub) : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)
  mo.output %norm_quant#0, %norm_quant#1 : !mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>
}
// CHECK-LABEL: mo.graph @rms_norm_quantize_dynamic_scaled_fp8_group_size_missing<total_seq_len>
// CHECK: mo.reduce.rms_norm
// CHECK: "mo.quantize_dynamic_scaled_float8"

mo.graph @rms_norm_quantize_dynamic_scaled_fp8_extra_parameters<total_seq_len>(
  %input: !mo.tensor<[total_seq_len, 10], bf16, gpu:0>,
  %gamma: !mo.tensor<[10], bf16, gpu:0>
) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>) {
  %norm = mo.reduce.rms_norm(%input, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[10], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 10], bf16, gpu:0>
  %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>
  %norm_quant:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 0>, parameters = {group_size_or_per_token = 10 : index, ultra_quantize = true}](%norm, %scale_ub) : (!mo.tensor<[total_seq_len, 10], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)
  mo.output %norm_quant#0, %norm_quant#1 : !mo.tensor<[total_seq_len, 10], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>
}
// CHECK-LABEL: mo.graph @rms_norm_quantize_dynamic_scaled_fp8_extra_parameters<total_seq_len>
// CHECK: mo.reduce.rms_norm
// CHECK: "mo.quantize_dynamic_scaled_float8"
