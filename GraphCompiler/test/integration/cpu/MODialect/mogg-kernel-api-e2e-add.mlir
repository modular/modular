// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=2 --input-shapes=2 --model-inputs=ones | FileCheck %s

// CHECK: 'dynamic' returned tensor<2xf32> [8.000000e+00, 8.000000e+00]
mo.graph @dynamic<D0>(%x: !mo.tensor<[D0], f32>, %y: !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
    %cst = mo.constant <#M.dense_array<4.> : tensor<1xf32>> : !mo.tensor<[1], f32>
    %shape = mo.shape_of(%x) : (!mo.tensor<[D0], f32>) -> !mo.tensor<[1], si64>
    %bcast = mo.broadcast_to(%cst, %shape) : (!mo.tensor<[1], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[D0], f32>
    %z = mo.custom [symbol = "imposter_add", device = #M.device_ref<"cpu", 0>] (%x, %y) : (!mo.tensor<[D0], f32>, !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32>
    %out = mo.mul(%z, %bcast) : !mo.tensor<[D0], f32>
    mo.output %out : !mo.tensor<[D0], f32>
}
