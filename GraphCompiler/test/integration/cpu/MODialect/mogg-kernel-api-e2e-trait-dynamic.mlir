// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=2x3 --model-inputs=ones | FileCheck %s

// A role-bearing trait argument with shape `[2, D0]` requires the capture's
// inferred layout type to retain both static and runtime extents.

// CHECK: 'trait_negate_dynamic' returned tensor<2x3xf32> [-1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00]
mo.graph @trait_negate_dynamic<D0>(%x: !mo.tensor<[2, D0], f32>) -> !mo.tensor<[2, D0], f32> {
    %z = mo.custom [symbol = "trait_negate", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[2, D0], f32>) -> !mo.tensor<[2, D0], f32>
    mo.output %z : !mo.tensor<[2, D0], f32>
}
