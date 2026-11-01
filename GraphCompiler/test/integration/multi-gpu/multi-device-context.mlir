// RUN: mo-opt --gc-pipeline="compiled-framework-label=mof kernel-library-paths=%kernel_api_test_lib" %s -o %t
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=compact %t | FileCheck %s

// CHECK-LABEL: --- Running 'multi_variadic_device_context':
// CHECK: dev_ctxs[ 0 ].id() = 0
// CHECK-NEXT: dev_ctxs[ 1 ].id() = 1

mo.graph @multi_variadic_device_context(
    %arg0: !mo.tensor<[4], f32>
) -> !mo.tensor<[4], f32> {
  %ch0 = mo.chain.create()
  %0, %ch1 = mo.transfer[%ch0] %arg0 : !mo.tensor<[4], f32> to <"gpu", 0>
  %1, %ch2 = mo.transfer[%ch1] %arg0 : !mo.tensor<[4], f32> to <"gpu", 1>
  %2 = mo.custom [symbol = "variadic_device_context", device = #M.device_ref<"cpu", 0>](
    %0, %1
  ) : (
    !mo.tensor<[4], f32, gpu:0>,
    !mo.tensor<[4], f32, gpu:1>
  ) -> !mo.tensor<[4], f32, gpu:0>
  %3, %ch3 = mo.transfer[%ch2] %2 : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
  mo.output %3 : !mo.tensor<[4], f32>
}
