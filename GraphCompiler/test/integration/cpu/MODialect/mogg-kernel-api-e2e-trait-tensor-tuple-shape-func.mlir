// RUN: mo-opt --gc-pipeline=kernel-library-paths="%kernel_api_test_pkg" %s -o %t.mlir
// RUN: mt --execute --input-shapes=2x3 --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// A shape function can take a variadic operand list as a `TensorTuple`. The
// custom op `trait_tensor_tuple_shape_op` registers
// `trait_tensor_tuple_shape_op_shape[Ts: TypeList[Trait=Tensor, ...]]`, so
// each operand is emitted as a `_MoggShapeTensorArg` that the tuple holds by
// value. The input shape is dynamic, so the shape function runs at runtime
// (a `mogg.call.shape`) instead of folding into the MGP init phase. The
// execute entry stays on the legacy lane and copies the first input.

// CHECK-LABEL: --- Running 'trait_tensor_tuple_shape'
// CHECK: 'trait_tensor_tuple_shape' returned tensor<2x3xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
mo.graph @trait_tensor_tuple_shape<D0, D1 -> D2, D3>(
  %arg0: !mo.tensor<[D0, D1], f32>
) -> !mo.tensor<[D2, D3], f32> {
  %res = mo.custom<() -> D4, D5> [symbol = "trait_tensor_tuple_shape_op", device = #M.device_ref<"cpu", 0>](%arg0, %arg0) :
    (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}
