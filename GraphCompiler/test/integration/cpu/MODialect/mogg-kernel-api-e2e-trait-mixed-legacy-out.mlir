// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --input-shapes=4 --model-inputs=ones | FileCheck %s

// End-to-end proof of a MIXED-LANE kernel with the lanes swapped from
// `trait_mixed_legacy_add`: `trait_mixed_legacy_out_add`'s output is a legacy
// `OutputTensor` (`ManagedTensorSlice`), while its two inputs carry the
// composable tensor-argument traits (`Some[Input & TileTensorable]`). Both the
// legacy write path (`OutputTensor.__setitem__`) and the trait read path
// (`to_tile_tensor(...)` loads) fire in the same emitted kernel. With
// all-ones inputs, `z[i] = x[i] + y[i]` must return all 2.

// CHECK: 'trait_mixed_legacy_out_add_graph' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @trait_mixed_legacy_out_add_graph(%x: !mo.tensor<[4], f32>, %y: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %z = mo.custom [symbol = "trait_mixed_legacy_out_add", device = #M.device_ref<"cpu", 0>] (%x, %y) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
