// RUN: mo-opt %s -o %t.mlir --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// This kernel simply copies the input to the output and adds the bias to the
// first term only.
//
// CHECK: 'variadic_input_to_output_size3_f32' returned
// CHECK-SAME: tensor<6xf32> [1.000100e+04, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]
// CHECK-NEXT: tensor<6xf32> [1.001000e+04, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]
// CHECK-NEXT: tensor<6xf32> [1.010000e+04, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]
mo.graph @variadic_input_to_output_size3_f32() -> (!mo.tensor<[6], f32>, !mo.tensor<[6], f32>, !mo.tensor<[6], f32>) {
  %in1 = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %in2 = mo.constant <#M.dense_array<10.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %in3 = mo.constant <#M.dense_array<100.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %bias = mo.constant <#M.dense_array<10000.0> : tensor<1xf32>> : !mo.tensor<[1], f32>

  %res1, %res2, %res3 = mo.custom [symbol = "variadic_input_to_output", device = #M.device_ref<"cpu", 0>] (%bias, %in1, %in2, %in3) : (
    !mo.tensor<[1], f32>,
    !mo.tensor<[6], f32>,
    !mo.tensor<[6], f32>,
    !mo.tensor<[6], f32>
  ) -> (!mo.tensor<[6], f32>, !mo.tensor<[6], f32>, !mo.tensor<[6], f32>)
  mo.output %res1, %res2, %res3 : !mo.tensor<[6], f32>, !mo.tensor<[6], f32>, !mo.tensor<[6], f32>
}

// CHECK: 'variadic_input_to_output_size4_si32' returned
// CHECK-SAME: tensor<6xsi32> [10001, 2, 3, 4, 5, 6]
// CHECK-NEXT: tensor<6xsi32> [10010, 2, 3, 4, 5, 6]
// CHECK-NEXT: tensor<6xsi32> [10100, 2, 3, 4, 5, 6]
// CHECK-NEXT: tensor<6xsi32> [11000, 2, 3, 4, 5, 6]
mo.graph @variadic_input_to_output_size4_si32() -> (!mo.tensor<[6], si32>, !mo.tensor<[6], si32>, !mo.tensor<[6], si32>, !mo.tensor<[6], si32>) {
  %in1 = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %in2 = mo.constant <#M.dense_array<10, 2, 3, 4, 5, 6> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %in3 = mo.constant <#M.dense_array<100, 2, 3, 4, 5, 6> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %in4 = mo.constant <#M.dense_array<1000, 2, 3, 4, 5, 6> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %bias = mo.constant <#M.dense_array<10000> : tensor<1xsi32>> : !mo.tensor<[1], si32>

  %res1, %res2, %res3, %res4 = mo.custom [symbol = "variadic_input_to_output", device = #M.device_ref<"cpu", 0>] (%bias, %in1, %in2, %in3, %in4) : (
    !mo.tensor<[1], si32>,
    !mo.tensor<[6], si32>,
    !mo.tensor<[6], si32>,
    !mo.tensor<[6], si32>,
    !mo.tensor<[6], si32>
  ) -> (!mo.tensor<[6], si32>, !mo.tensor<[6], si32>, !mo.tensor<[6], si32>, !mo.tensor<[6], si32>)
  mo.output %res1, %res2, %res3, %res4 : !mo.tensor<[6], si32>, !mo.tensor<[6], si32>, !mo.tensor<[6], si32>, !mo.tensor<[6], si32>
}
