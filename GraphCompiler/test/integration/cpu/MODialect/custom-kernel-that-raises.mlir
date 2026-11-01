// RUN: mo-opt --gc-pipeline=kernel-library-paths="%kernel_api_test_pkg" %s -o %t.mlir
// RUN: not mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @custom_success(%arg0: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %0 = mo.custom [symbol = "custom_op_that_raises", device = #M.device_ref<"cpu", 0>](%arg0) : (!mo.tensor<[2], f32>) -> !mo.tensor<[2], f32>
  mo.output %0 : !mo.tensor<[2], f32>
}

// CHECK-LABEL: --- Running 'custom_success':
// CHECK-NEXT: 'custom_success' returned tensor<2xf32> [1.000000e+00, 1.000000e+00]

mo.graph @custom_raises(%arg0: !mo.tensor<[10], f32>) -> !mo.tensor<[10], f32> {
  %0 = mo.custom [symbol = "custom_op_that_raises", device = #M.device_ref<"cpu", 0>](%arg0) : (!mo.tensor<[10], f32>) -> !mo.tensor<[10], f32>
  mo.output %0 : !mo.tensor<[10], f32>
}

// CHECK-LABEL: --- Running 'custom_raises':
// CHECK: 'custom_raises' returned <error:
// CHECK: input_shape[0] == 10
