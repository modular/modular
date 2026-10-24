// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir --input-shapes=1x2x3 | FileCheck %s

// RUN: mo-opt %s -o %t_mojo.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t_mojo.mlir --input-shapes=1x2x3 | FileCheck %s

// CHECK-LABEL: Running 'print_shape':
// CHECK: input.shape = (1, 2, 3)
// CHECK: output.shape = (1, 2, 3)
mo.graph @print_shape<D0, D1>(%0: !mo.tensor<[D0, D1, 3], f32>) -> (!mo.tensor<[D0, D1, 3], f32>) {
  %1 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>] (%0, %0) : (!mo.tensor<[D0, D1, 3], f32>, !mo.tensor<[D0, D1, 3], f32>) -> !mo.tensor<[D0, D1, 3], f32>
  %2 = mo.custom [symbol = "print_shape_fused", device = #M.device_ref<"cpu", 0>] (%1) : (!mo.tensor<[D0, D1, 3], f32>) -> !mo.tensor<[D0, D1, 3], f32>
  mo.output %2: !mo.tensor<[D0, D1, 3], f32>
}
