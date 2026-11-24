// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=10x20 --input-shapes=20x30 --model-inputs=random | FileCheck %s

// Golden output generated from disabling the pattern.
// Regex `[0-9\-\.e ,]+` checks against fp strings like
// "-9.999551e-01, -8.299351e-01, 2.027053e-01 ..."
// CHECK: --- Running 'ndbuffer_matmul':
// CHECK-NEXT: returned tensor<10x30xf32> [
// CHECK-SAME: [[RESULT:[0-9\.e ,\-]+]]
// CHECK-SAME: ]
mo.graph @ndbuffer_matmul<D0, D1, D2>(%x: !mo.tensor<[D0, D1], f32>, %y: !mo.tensor<[D1, D2], f32>) -> !mo.tensor<[D0, D2], f32> {
    %z = mo.matmul(%x, %y) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D1, D2], f32>) -> !mo.tensor<[D0, D2], f32>
    mo.output %z : !mo.tensor<[D0, D2], f32>
}

// CHECK: --- Running 'extensibility_matmul':
// CHECK-NEXT: returned tensor<10x30xf32> [
// CHECK-SAME: [[RESULT]]
// CHECK-SAME: ]
mo.graph @extensibility_matmul<D0, D1, D2>(%x: !mo.tensor<[D0, D1], f32>, %y: !mo.tensor<[D1, D2], f32>) -> !mo.tensor<[D0, D2], f32> {
    %z = mo.custom [symbol = "imposter_matmul", device = #M.device_ref<"cpu", 0>] (%x, %y) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D1, D2], f32>) -> !mo.tensor<[D0, D2], f32>
    mo.output %z : !mo.tensor<[D0, D2], f32>
}
