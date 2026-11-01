// RUN: not mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" 2>&1 | FileCheck %s

// Negative: a rank-0 f32 attribute against a kernel parameter that
// declares as an integer SIMD (`Int32` == `SIMD[int32, 1]`) is not
// rejected by parameter verification (that path keys on the SIMD kind
// and only checks the f32 spelling), so it surfaces at Mojo elaboration,
// where binding the float to the integer parameter fails. Pin that
// failure so the case stays loud instead of silently regressing to a
// successful compile.
//
// CHECK: error: function instantiation failed
// CHECK: constraint failed: the SIMD type must be floating point
mo.graph @test_int_simd_float_fail(%arg0: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  %res = mo.custom [symbol = "op_with_custom_params", device = #M.device_ref<"cpu", 0>, parameters = { "custom_int" = 42, "custom_i32" = #M.dense_array<2.5> : tensor<f32>, "custom_str" = "hello", "custom_dtype" = si32, "custom_float" = #M.dense_array<2.5> : tensor<f32> }] (%arg0) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  mo.output %res: !mo.tensor<[4], f32>
}
