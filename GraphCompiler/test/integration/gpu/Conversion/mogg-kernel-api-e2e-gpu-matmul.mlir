// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=10x20 --input-shapes=20x30 --model-inputs=random | FileCheck %s

// Golden output generated from disabling the pattern.
// Regex `[0-9\-\.e ,]+` checks against fp strings like
// "-9.999551e-01, -8.299351e-01, 2.027053e-01 ..."
// CHECK: --- Running 'ndbuffer_matmul':
// CHECK-NEXT: returned tensor<10x30xf32> [
// CHECK-SAME: [[RESULT:[0-9\.e ,\-]+]]
// CHECK-SAME: ]
mo.graph @ndbuffer_matmul(%x: !mo.tensor<[?, ?], f32>, %y: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[?, ?], f32> to <"gpu", 0>
    %y_gpu, %ch2 = mo.transfer[%ch1] %y : !mo.tensor<[?, ?], f32> to <"gpu", 0>
    %z_gpu = mo.matmul(%x_gpu, %y_gpu) { device = #M.device_ref<"gpu", 0> } : (!mo.tensor<[?, ?], f32, gpu:0>, !mo.tensor<[?, ?], f32, gpu:0>) -> !mo.tensor<[?, ?], f32, gpu:0>
    %z, %ch3 = mo.transfer[%ch2] %z_gpu : !mo.tensor<[?, ?], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[?, ?], f32>
}

// CHECK: --- Running 'extensibility_matmul':
// CHECK-NEXT: returned tensor<10x30xf32> [
// CHECK-SAME: [[RESULT]]
// CHECK-SAME: ]
mo.graph @extensibility_matmul(%x: !mo.tensor<[?, ?], f32>, %y: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32> {
    %z = mo.custom [symbol = "imposter_matmul", device = #M.device_ref<"gpu", 0>] (%x, %y) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32>
    mo.output %z : !mo.tensor<[?, ?], f32>
}
