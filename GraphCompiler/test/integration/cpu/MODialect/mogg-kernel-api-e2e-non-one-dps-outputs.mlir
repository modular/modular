// RUN: mo-opt %s -o %t.mlir --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// Check kernels with non-1 dps outputs work as expected
// CHECK: 'test_kernel_with_2_dps_outputs' returned tensor<6xsi32> [101,
// CHECK-NEXT: tensor<6xsi32> [110,
mo.graph @test_kernel_with_2_dps_outputs() -> (!mo.tensor<[6], si32>, !mo.tensor<[6], si32>) {
  %const1 = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %const2 = mo.constant <#M.dense_array<10, 2, 3, 4, 5, 6> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %bias = mo.constant <#M.dense_array<100, 2, 3, 4, 5, 6> : tensor<6xsi32>> : !mo.tensor<[6], si32>

  // %out1[0] = %const1[0] + %bias[0]
  // %out2[0] = %const2[0] + %bias[0]
  %out1, %out2 = mo.custom [symbol = "add_bias_to_two_tensors", device = #M.device_ref<"cpu", 0>] (%const1, %const2, %bias) : (
    !mo.tensor<[6], si32>,
    !mo.tensor<[6], si32>,
    !mo.tensor<[6], si32>
  ) -> (!mo.tensor<[6], si32>, !mo.tensor<[6], si32>)

  mo.output %out1, %out2: !mo.tensor<[6], si32>, !mo.tensor<[6], si32>
}
