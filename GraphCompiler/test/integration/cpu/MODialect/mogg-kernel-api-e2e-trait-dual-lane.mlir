// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// One value, two lanes, end to end: `trait_dual_lane_add` declares a trait
// dense input NEXT TO a legacy `InputTensor` input, and the graph passes the
// same tensor to both. `SimplifyKernelOperands` merges the duplicates into
// one exec-body argument, so the stub must hand the trait position a
// `_MoggDenseTensorArg` and the legacy position a `ManagedTensorSlice`, both
// over the same buffer.
//
// Related to GEX-4071.

// CHECK: 'trait_dual_lane_graph' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @trait_dual_lane_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %z = mo.custom [symbol = "trait_dual_lane_add", device = #M.device_ref<"cpu", 0>] (%x, %x) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
