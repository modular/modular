// RUN: mo-opt --gc-pipeline=kernel-library-paths="%kernel_api_test_pkg" %s -o %t.mlir
// RUN: mt --execute --input-shapes=2x3 --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// End-to-end proof of the trait-typed shape-function emit path. The custom op
// `trait_metadata_shape_op` registers a metadata-only, no-role trait shape
// function (`trait_metadata_shape_op_shape[T: TileTensorable](input: T)`), so
// its tensor operand is emitted as a `_MoggShapeTensorArg` (immutable, host,
// data-visible) rather than rebound to a legacy
// `InputTensor`, and the call result is rebound to the concrete
// `IndexList[output_rank]`. The execute entry stays on the legacy lane and
// copies input -> output.
//
// The input shape is passed dynamically (`--input-shapes=2x3`) so the shape
// function is invoked at *runtime* (a `mogg.call.shape`), not folded into the
// MGP init phase -- this is the path the trait emit change affects. With an
// all-ones 2x3 input the copy must return a 2x3 tensor of ones, proving the
// emitted Mojo compiles and the runtime shape call produces the right shape.

// CHECK-LABEL: --- Running 'trait_metadata_shape'
// CHECK: 'trait_metadata_shape' returned tensor<2x3xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
mo.graph @trait_metadata_shape<D0, D1 -> D2, D3>(
  %arg0: !mo.tensor<[D0, D1], f32>
) -> !mo.tensor<[D2, D3], f32> {
  %res = mo.custom<() -> D4, D5> [symbol = "trait_metadata_shape_op", device = #M.device_ref<"cpu", 0>](%arg0) :
    (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}
