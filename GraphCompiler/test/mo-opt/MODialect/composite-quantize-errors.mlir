// RUN: mo-opt %s --split-input-file --verify-diagnostics

// Each case isolates one QuantizeDynamicScaledFP8Op::verify() rejection.
// Valid ops live in the RMS-norm and all-reduce fusion suites.

mo.graph @rank3_input(%x: !mo.tensor<[2, 4, 8], bf16, gpu:0>, %ub: !mo.tensor<[], f32>) -> !mo.tensor<[2, 4, 8], f8e4m3fn, gpu:0> {
  // expected-error@below {{'mo.composite.quantize_dynamic_scaled_fp8' op the input must be [rows, cols]}}
  %q:2 = mo.composite.quantize_dynamic_scaled_fp8(%x, %ub) {group_size_or_per_token = -1} : (!mo.tensor<[2, 4, 8], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[2, 4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[1, 8], bf16, gpu:0>)
  mo.output %q#0 : !mo.tensor<[2, 4, 8], f8e4m3fn, gpu:0>
}

// -----

mo.graph @output_shape(%x: !mo.tensor<[4, 8], bf16, gpu:0>, %ub: !mo.tensor<[], f32>) -> !mo.tensor<[4, 16], f8e4m3fn, gpu:0> {
  // expected-error@below {{'mo.composite.quantize_dynamic_scaled_fp8' op the output must have the same shape as the input}}
  %q:2 = mo.composite.quantize_dynamic_scaled_fp8(%x, %ub) {group_size_or_per_token = -1} : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[4, 16], f8e4m3fn, gpu:0>, !mo.tensor<[1, 4], bf16, gpu:0>)
  mo.output %q#0 : !mo.tensor<[4, 16], f8e4m3fn, gpu:0>
}

// -----

mo.graph @rank1_scales(%x: !mo.tensor<[4, 8], bf16, gpu:0>, %ub: !mo.tensor<[], f32>) -> !mo.tensor<[4], bf16, gpu:0> {
  // expected-error@below {{'mo.composite.quantize_dynamic_scaled_fp8' op the scales must be rank 2}}
  %q:2 = mo.composite.quantize_dynamic_scaled_fp8(%x, %ub) {group_size_or_per_token = -1} : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4], bf16, gpu:0>)
  mo.output %q#1 : !mo.tensor<[4], bf16, gpu:0>
}

// -----

mo.graph @zero_group_size(%x: !mo.tensor<[4, 8], bf16, gpu:0>, %ub: !mo.tensor<[], f32>) -> !mo.tensor<[4, 8], f8e4m3fn, gpu:0> {
  // expected-error@below {{'mo.composite.quantize_dynamic_scaled_fp8' op group_size_or_per_token must be -1 or positive}}
  %q:2 = mo.composite.quantize_dynamic_scaled_fp8(%x, %ub) {group_size_or_per_token = 0} : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[1, 4], bf16, gpu:0>)
  mo.output %q#0 : !mo.tensor<[4, 8], f8e4m3fn, gpu:0>
}

// -----

mo.graph @negative_amax_floor(%x: !mo.tensor<[4, 8], bf16, gpu:0>, %ub: !mo.tensor<[], f32>) -> !mo.tensor<[4, 8], f8e4m3fn, gpu:0> {
  // expected-error@below {{'mo.composite.quantize_dynamic_scaled_fp8' op amax_floor must be finite and non-negative}}
  %q:2 = mo.composite.quantize_dynamic_scaled_fp8(%x, %ub) {group_size_or_per_token = -1, amax_floor = #M.dense_array<-1.0> : tensor<f32>} : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[1, 4], bf16, gpu:0>)
  mo.output %q#0 : !mo.tensor<[4, 8], f8e4m3fn, gpu:0>
}
