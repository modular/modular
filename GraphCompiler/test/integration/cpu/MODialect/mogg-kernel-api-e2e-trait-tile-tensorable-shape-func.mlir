// RUN: mo-opt --gc-pipeline=kernel-library-paths="%kernel_api_test_pkg" %s -o %t.mlir
// RUN: mt --execute --input-shapes=2x2 --input-shapes=2 --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// End-to-end proof of the trait-typed shape-function emit path for the
// `TileTensorable` tier reading actual element values. The custom op
// `trait_reshape_shape_op` registers a shape function
// (`trait_reshape_shape_op_shape[D: Tensor, S: TileTensorable](data: D,
// sizes: S)`) that reads `sizes`'s runtime element values via
// `to_tile_tensor()` to build the output `IndexList`, rather than deriving it
// from `data`'s own shape. `data` mixes in the bare `Tensor` bound, so one
// shape function exercises two tiers at once.
//
// Both `data` (dynamic 2x2) and `sizes` are genuine graph block arguments --
// neither is a `mo.constant` -- so the shape function executes at *runtime*
// (a `mogg.call.shape`) rather than being folded during the MGP init phase,
// proving the emitted Mojo actually reads `sizes`'s buffer at call time. With
// all-ones inputs, `sizes` reads as `[1, 1]`, so the output shape collapses
// to 1x1 even though `data` is 2x2; `execute` copies `data`'s leading element
// into that output, giving a verifiable 1x1 result.

// CHECK-LABEL: --- Running 'trait_reshape_shape'
// CHECK: 'trait_reshape_shape' returned tensor<1x1xf32> [1.000000e+00]
mo.graph @trait_reshape_shape<D0, D1 -> D2, D3>(
  %data: !mo.tensor<[D0, D1], f32>,
  %sizes: !mo.tensor<[2], si64>
) -> !mo.tensor<[D2, D3], f32> {
  %res = mo.custom<() -> D4, D5> [symbol = "trait_reshape_shape_op", device = #M.device_ref<"cpu", 0>](%data, %sizes) :
    (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}
