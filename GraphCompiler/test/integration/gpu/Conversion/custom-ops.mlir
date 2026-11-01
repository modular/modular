// RUN: mo-opt --gc-pipeline="compiled-framework-label=mof constant-fold=false default-device-label=gpu kernel-library-paths=%kernel_api_test_lib" %s -o %t
// RUN: mt --execute --default-device-label=gpu --soft-fusion-non-host-kernel-limit=3 --model-inputs=ones --result-output-style=compact %t | FileCheck %s

// CHECK-LABEL: --- Running 'custom_op_test':
// CHECK: hello from kernel on gpu
// CHECK-NEXT: returned tensor

mo.graph @custom_op_test(%a : !mo.tensor<[5000, 2], f32>) -> !mo.tensor<[5000, 2], f32> {
  %0 = mo.custom [symbol = "basic_target", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[5000, 2], f32>) -> !mo.tensor<[5000, 2], f32>
  mo.output %0 : !mo.tensor<[5000, 2], f32 >
}
