// RUN: mo-opt %s -o %t.mlir --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// CHECK-LABEL: Running 'elementwise1':
// CHECK: 'elementwise1' returned tensor<2x3xf32> [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]
mo.graph @elementwise1() -> !mo.tensor<[2, 3], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00, 3.000000e+00, 3.500000e+00> : tensor<6xf32>> : !mo.tensor<[2, 3], f32>
  %1 = mo.constant <#M.dense_array<3.000000e+00, 3.500000e+00, 4.000000e+00, 4.500000e+00, 5.000000e+00, 5.500000e+00> : tensor<6xf32>> : !mo.tensor<[2, 3], f32>
  %2 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%0, %1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %2 : !mo.tensor<[2, 3], f32>
}

// CHECK-LABEL: Running 'elementwise3':
// CHECK: 'elementwise3' returned tensor<2x3xf32> [1.000000e+01, 1.200000e+01, 1.400000e+01, 1.600000e+01, 1.800000e+01, 2.000000e+01]
mo.graph @elementwise3() -> !mo.tensor<[2, 3], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00, 3.000000e+00, 3.500000e+00> : tensor<6xf32>> : !mo.tensor<[2, 3], f32>
  %1 = mo.constant <#M.dense_array<3.000000e+00, 3.500000e+00, 4.000000e+00, 4.500000e+00, 5.000000e+00, 5.500000e+00> : tensor<6xf32>> : !mo.tensor<[2, 3], f32>
  %2 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : tensor<6xf32>> : !mo.tensor<[2, 3], f32>
  %3 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%0, %1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  %4 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%3, %2) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  %5 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%4, %4) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %5 : !mo.tensor<[2, 3], f32>
}

// CHECK: has_epilogue_fusion = True
// CHECK-LABEL: Running 'matmul_add':
// CHECK: 'matmul_add' returned tensor<2x2xf32> [6.500000e+00, 5.500000e+00, 1.350000e+01, 1.250000e+01]
mo.graph @matmul_add() -> !mo.tensor<[2, 2], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : tensor<4xf32>> : !mo.tensor<[2, 2], f32>
  %1 = mo.constant <#M.dense_array<5.000000e-01, 2.000000e+00, 2.500000e+00, 1.000000e+00> : tensor<4xf32>> : !mo.tensor<[2, 2], f32>
  %2 = mo.constant <#M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00> : tensor<4xf32>> : !mo.tensor<[2, 2], f32>
  %3 = mo.custom [symbol = "matmul_fuse_out", device = #M.device_ref<"cpu", 0>](%0, %1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %4 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%3, %2) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %4 : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: Running 'broadcast_add':
// CHECK: 'broadcast_add' returned tensor<3x4xf32> [2.000000e+00, 3.500000e+00, 5.000000e+00, 6.500000e+00, 4.000000e+00, 5.500000e+00, 7.000000e+00, 8.500000e+00, 6.000000e+00, 7.500000e+00, 9.000000e+00, 1.050000e+01]
mo.graph @broadcast_add() -> !mo.tensor<[3, 4], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %1 = mo.constant <#M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00, 3.000000e+00, 3.500000e+00, 4.000000e+00, 4.500000e+00, 5.000000e+00, 5.500000e+00, 6.000000e+00, 6.500000e+00> : tensor<12xf32>> : !mo.tensor<[3, 4], f32>
  %2 = mo.static.broadcast_to(%0) : !mo.tensor<[4], f32> -> !mo.tensor<[3, 4], f32>
  %3 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%2, %1) : (!mo.tensor<[3, 4], f32>, !mo.tensor<[3, 4], f32>) -> !mo.tensor<[3, 4], f32>
  mo.output %3 : !mo.tensor<[3, 4], f32>
}

// CHECK-LABEL: Running 'transpose_add':
// CHECK: 'transpose_add' returned tensor<3x4xf32> [1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00, 3.000000e+00, 3.500000e+00, 4.000000e+00, 4.500000e+00, 5.000000e+00, 5.500000e+00, 6.000000e+00, 6.500000e+00]
mo.graph @transpose_add(%arg0: !mo.tensor<[4, 3], f32>) -> !mo.tensor<[3, 4], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00, 3.000000e+00, 3.500000e+00, 4.000000e+00, 4.500000e+00, 5.000000e+00, 5.500000e+00, 6.000000e+00, 6.500000e+00> : tensor<3x4xf32>> : !mo.tensor<[3, 4], f32>
  %1 = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.transpose(%arg0, %1) : (!mo.tensor<[4, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 4], f32>
  %3 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%2, %0) : (!mo.tensor<[3, 4], f32>, !mo.tensor<[3, 4], f32>) -> !mo.tensor<[3, 4], f32>
  mo.output %3 : !mo.tensor<[3, 4], f32>
}

// matmul (not fused)
// CHECK: has_epilogue_fusion = False
// CHECK-LABEL: Running 'matmul_no_fusion':
// CHECK: 'matmul_no_fusion' returned tensor<2x2xf32> [5.500000e+00, 4.000000e+00, 1.150000e+01, 1.000000e+01]
mo.graph @matmul_no_fusion() -> !mo.tensor<[2, 2], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : tensor<4xf32>> : !mo.tensor<[2, 2], f32>
  %1 = mo.constant <#M.dense_array<5.000000e-01, 2.000000e+00, 2.500000e+00, 1.000000e+00> : tensor<4xf32>> : !mo.tensor<[2, 2], f32>
  %2 = mo.custom [symbol = "matmul_fuse_out", device = #M.device_ref<"cpu", 0>](%0, %1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %2 : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: Running 'elemwise_input_fusion':
// CHECK: 'elemwise_input_fusion' returned tensor<4xf32> [3.500000e+00, 5.500000e+00, 7.500000e+00, 9.500000e+00]
mo.graph @elemwise_input_fusion() -> !mo.tensor<[4], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %1 = mo.constant <#M.dense_array<2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %2 = mo.constant <#M.dense_array<5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %3 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%0, %1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %4 = mo.custom [symbol = "imposter_add_lhs", device = #M.device_ref<"cpu", 0>](%3, %2) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %4 : !mo.tensor<[4], f32>
}

// CHECK-LABEL: Running 'elemwise_input_fusion_dbl':
// CHECK: 'elemwise_input_fusion_dbl' returned tensor<4xf32> [6.500000e+00, 1.000000e+01, 1.350000e+01, 1.700000e+01]
mo.graph @elemwise_input_fusion_dbl() -> !mo.tensor<[4], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %1 = mo.constant <#M.dense_array<2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %2 = mo.constant <#M.dense_array<5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %3 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%0, %1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %4 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%3, %3) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %5 = mo.custom [symbol = "imposter_add_lhs", device = #M.device_ref<"cpu", 0>](%4, %2) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %5 : !mo.tensor<[4], f32>
}

// CHECK-LABEL: Running 'elemwise_input_fusion_multi_inputs':
// CHECK: 'elemwise_input_fusion_multi_inputs' returned tensor<4xf32> [3.500000e+00, 5.500000e+00, 7.500000e+00, 9.500000e+00]
mo.graph @elemwise_input_fusion_multi_inputs() -> !mo.tensor<[4], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %1 = mo.constant <#M.dense_array<2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %2 = mo.constant <#M.dense_array<1.000000e+00, 5.000000e-01, 1.250000e+00, 1.500000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %3 = mo.constant <#M.dense_array<-5.000000e-01, 5.000000e-01, 2.500000e-01, 5.000000e-01> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %4 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%2, %3) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %5 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%0, %1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %6 = mo.custom [symbol = "imposter_add_fuse_inputs", device = #M.device_ref<"cpu", 0>](%4, %5) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %6 : !mo.tensor<[4], f32>
}

// CHECK-LABEL: Running 'variadic_fusion':
// CHECK: 'variadic_fusion' returned tensor<4xf32> [3.500000e+00, 5.500000e+00, 7.500000e+00, 9.500000e+00]
mo.graph @variadic_fusion() -> !mo.tensor<[4], f32> {
  %0 = mo.constant <#M.dense_array<1.500000e+00, 1.250000e+00, 1.000000e+00, 2.000000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %1 = mo.constant <#M.dense_array<-5.000000e-01, 2.500000e-01, 1.000000e+00, 5.000000e-01> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %2 = mo.constant <#M.dense_array<2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %3 = mo.constant <#M.dense_array<5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %4 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%0, %1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %5 = mo.custom [symbol = "variadic_add", device = #M.device_ref<"cpu", 0>](%4, %2, %3) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %5 : !mo.tensor<[4], f32>
}

// CHECK-LABEL: Running 'variadic_fusion_multiple':
// CHECK: 'variadic_fusion_multiple' returned tensor<4xf32> [3.500000e+00, 5.500000e+00, 7.500000e+00, 9.500000e+00]
mo.graph @variadic_fusion_multiple() -> !mo.tensor<[4], f32> {
  %0 = mo.constant <#M.dense_array<1.500000e+00, 1.250000e+00, 1.000000e+00, 2.000000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %1 = mo.constant <#M.dense_array<-5.000000e-01, 2.500000e-01, 1.000000e+00, 5.000000e-01> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %2 = mo.constant <#M.dense_array<2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %3 = mo.constant <#M.dense_array<1.500000e+00, -5.000000e-01, 1.000000e+00, 5.000000e-01> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %4 = mo.constant <#M.dense_array<-1.000000e+00, 1.500000e+00, 5.000000e-01, 1.500000e+00> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %5 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%0, %1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %6 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%3, %4) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %7 = mo.custom [symbol = "variadic_add", device = #M.device_ref<"cpu", 0>](%5, %2, %6) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %7 : !mo.tensor<[4], f32>
}
