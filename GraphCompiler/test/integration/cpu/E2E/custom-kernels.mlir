// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir| FileCheck %s

// CHECK-LABEL: Running 'print_shape':
// CHECK: input.shape = (1, 2, 3)
// CHECK: output.shape = (1, 2, 3)
mo.graph @print_shape(%0: !mo.tensor<[1, 2, 3], f32>) -> (!mo.tensor<[1, 2, 3], f32>) {
  %1 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>] (%0, %0) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32>
  %2 = mo.custom [symbol = "print_shape_fused", device = #M.device_ref<"cpu", 0>] (%1) : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32>
  mo.output %2: !mo.tensor<[1, 2, 3], f32>
}
