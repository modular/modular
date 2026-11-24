// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --model-inputs=ones | FileCheck %s

// A read-only trait argument loaded from mutable storage must remain on the
// trait assembly path rather than the legacy `ManagedTensorSlice` path.

// CHECK: 'trait_negate_from_mutable' returned tensor<4xf32> [-1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00]
mo.graph @trait_negate_from_mutable(%buf: !mo.buffer<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create(%buf) : !mo.buffer<[4], f32>
    %x, %ch1 = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[4], f32>, !mo.chain) -> (!mo.tensor<[4], f32>, !mo.chain)
    %z = mo.custom [symbol = "trait_negate", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
