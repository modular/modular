// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib"
// RUN: mt --execute %t.mlir --result-output-style=full --model-inputs=ones | FileCheck %s

// CHECK-LABEL: --- Running 'add_one_cpu':
// CHECK: AddOne CPU kernel
// CHECK: returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @add_one_cpu(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %out = mo.custom [symbol = "add_one", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %out : !mo.tensor<[4], f32>
}

// CHECK-LABEL: --- Running 'add_one_gpu':
// CHECK: AddOne GPU kernel
// CHECK: returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @add_one_gpu(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %out_gpu = mo.custom [symbol = "add_one", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %out, %ch2 = mo.transfer[%ch1] %out_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %out : !mo.tensor<[4], f32>
}
