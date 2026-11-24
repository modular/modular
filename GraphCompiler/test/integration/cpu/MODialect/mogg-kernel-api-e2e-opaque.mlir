// RUN: mo-opt %s -o %t.mlir --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// CHECK: Running 'test_kernel_with_0_dps_outputs_reg_types':
// CHECK-DAG: MyCustomScalarRegSI32.__init__ 100
// CHECK-DAG: MyCustomScalarRegSI32.__deinit__ 100
// CHECK-DAG: MyCustomScalarRegSI32.__init__ 42
// CHECK-DAG: MyCustomScalarRegSI32.__deinit__ 42
// CHECK: 'test_kernel_with_0_dps_outputs_reg_types' returned tensor<1xsi32> [142]
mo.graph @test_kernel_with_0_dps_outputs_reg_types() -> (!mo.tensor<[1], si32>) {
  %const1 = mo.constant <#M.dense_array<100> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %const2 = mo.constant <#M.dense_array<42> : tensor<1xsi32>> : !mo.tensor<[1], si32>

  // Returns a view of the input tensor of the same size, but all elements are the first element
  %opaque1 = mo.custom [symbol = "tensor_to_custom_scalar_si32_reg", device = #M.device_ref<"cpu", 0>] (%const1) : (!mo.tensor<[1], si32>) -> !mo.opaque<"MyCustomScalarRegSI32">
  %opaque2 = mo.custom [symbol = "tensor_to_custom_scalar_si32_reg", device = #M.device_ref<"cpu", 0>] (%const2) : (!mo.tensor<[1], si32>) -> !mo.opaque<"MyCustomScalarRegSI32">

  %res = mo.custom [symbol = "opaque_add_to_tensor_si32_reg", device = #M.device_ref<"cpu", 0>] (%opaque1, %opaque2) : (!mo.opaque<"MyCustomScalarRegSI32">, !mo.opaque<"MyCustomScalarRegSI32">) -> !mo.tensor<[1], si32>
  mo.output %res: !mo.tensor<[1], si32>
}

// CHECK: Running 'test_kernel_with_0_dps_outputs_mem_types':
// CHECK-DAG: MyCustomScalarSI32.__init__ 100
// CHECK-DAG: MyCustomScalarSI32.__deinit__ 100
// CHECK-DAG: MyCustomScalarSI32.__init__ 42
// CHECK-DAG: MyCustomScalarSI32.__deinit__ 42
// CHECK: 'test_kernel_with_0_dps_outputs_mem_types' returned tensor<1xsi32> [142]
mo.graph @test_kernel_with_0_dps_outputs_mem_types() -> (!mo.tensor<[1], si32>) {
  %const1 = mo.constant <#M.dense_array<100> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %const2 = mo.constant <#M.dense_array<42> : tensor<1xsi32>> : !mo.tensor<[1], si32>

  // Returns a view of the input tensor of the same size, but all elements are the first element
  %opaque1 = mo.custom [symbol = "tensor_to_custom_scalar_si32", device = #M.device_ref<"cpu", 0>] (%const1) : (!mo.tensor<[1], si32>) -> !mo.opaque<"MyCustomScalarSI32">
  %opaque2 = mo.custom [symbol = "tensor_to_custom_scalar_si32", device = #M.device_ref<"cpu", 0>] (%const2) : (!mo.tensor<[1], si32>) -> !mo.opaque<"MyCustomScalarSI32">

  %res = mo.custom [symbol = "opaque_add_to_tensor_si32", device = #M.device_ref<"cpu", 0>] (%opaque1, %opaque2) : (!mo.opaque<"MyCustomScalarSI32">, !mo.opaque<"MyCustomScalarSI32">) -> !mo.tensor<[1], si32>
  mo.output %res: !mo.tensor<[1], si32>
}

// Test throwing kernel with opaque memory types.
// This verifies that destructors are called correctly even when the kernel
// can throw, and that the next graph doesn't see stale destructor calls.
// The `opaque_add_to_tensor_si32_raises` kernel is marked as `raises`.
//
// CHECK: Running 'test_throwing_kernel_with_mem_types':
// CHECK-DAG: MyCustomScalarSI32.__init__ 200
// CHECK-DAG: MyCustomScalarSI32.__deinit__ 200
// CHECK-DAG: MyCustomScalarSI32.__init__ 58
// CHECK-DAG: MyCustomScalarSI32.__deinit__ 58
// CHECK: 'test_throwing_kernel_with_mem_types' returned tensor<1xsi32> [258]
mo.graph @test_throwing_kernel_with_mem_types() -> (!mo.tensor<[1], si32>) {
  %const1 = mo.constant <#M.dense_array<200> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %const2 = mo.constant <#M.dense_array<58> : tensor<1xsi32>> : !mo.tensor<[1], si32>

  %opaque1 = mo.custom [symbol = "tensor_to_custom_scalar_si32", device = #M.device_ref<"cpu", 0>] (%const1) : (!mo.tensor<[1], si32>) -> !mo.opaque<"MyCustomScalarSI32">
  %opaque2 = mo.custom [symbol = "tensor_to_custom_scalar_si32", device = #M.device_ref<"cpu", 0>] (%const2) : (!mo.tensor<[1], si32>) -> !mo.opaque<"MyCustomScalarSI32">

  // This kernel is marked as `raises` in testing_kernels.mojo
  %res = mo.custom [symbol = "opaque_add_to_tensor_si32_raises", device = #M.device_ref<"cpu", 0>] (%opaque1, %opaque2) : (!mo.opaque<"MyCustomScalarSI32">, !mo.opaque<"MyCustomScalarSI32">) -> !mo.tensor<[1], si32>
  mo.output %res: !mo.tensor<[1], si32>
}

// CHECK: Running 'test_kernel_with_dps_outputs_reg_trivial_types':
// CHECK: MyCustomScalarRegTrivialSI32.__init__ 100
// CHECK: 'test_kernel_with_dps_outputs_reg_trivial_types' returned <unknown>
mo.graph @test_kernel_with_dps_outputs_reg_trivial_types() -> (!mo.opaque<"MyCustomScalarRegTrivialSI32">) {
  %const1 = mo.constant <#M.dense_array<100> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %opaque1 = mo.custom [symbol = "tensor_to_custom_scalar_si32_reg_trivial", device = #M.device_ref<"cpu", 0>] (%const1) : (!mo.tensor<[1], si32>) -> !mo.opaque<"MyCustomScalarRegTrivialSI32">
  mo.output %opaque1: !mo.opaque<"MyCustomScalarRegTrivialSI32">
}
