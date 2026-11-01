// RUN: mo-opt %s -o %t.mlir --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// CHECK: Running 'test_op_with_custom_params':
// CHECK-NEXT: custom_int = 42
// CHECK-NEXT: custom_i32 = -7
// CHECK-NEXT: custom_str = hello
// CHECK-NEXT: custom_dtype = int32
// CHECK-NEXT: custom_float = 2.5
mo.graph @test_op_with_custom_params(%arg0: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  %res = mo.custom [symbol = "op_with_custom_params", device = #M.device_ref<"cpu", 0>, parameters = { "custom_int" = 42, "custom_i32" = -7, "custom_str" = "hello", "custom_dtype" = si32, "custom_float" = #M.dense_array<2.5> : tensor<f32> }] (%arg0) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  mo.output %res: !mo.tensor<[4], f32>
}

// CHECK: Running 'test_op_with_default_params':
// CHECK-NEXT: rank = 10
// CHECK-NEXT: rank_defaulted = 128
// CHECK-NEXT: rank_derived_from_other = 11
// CHECK-NEXT: rank_derived_from_derived = 12
mo.graph @test_op_with_default_params(%arg: !mo.tensor<[1], si32>) -> (!mo.tensor<[1], si32>) {
  %res = mo.custom [symbol = "kernel_with_default_parameter", device = #M.device_ref<"cpu", 0>, parameters = { "rank" = 10 }] (%arg) : (!mo.tensor<[1], si32>) -> !mo.tensor<[1], si32>

  mo.output %res: !mo.tensor<[1], si32>
}
