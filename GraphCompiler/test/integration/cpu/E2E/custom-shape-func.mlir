// RUN: mo-opt --gc-pipeline=kernel-library-paths="%kernel_api_test_pkg" %s -o %t.mlir
// RUN: not mt --execute --model-inputs=ones --input-shapes=2 --input-shapes=20 --result-output-style=full %t.mlir | FileCheck %s

// NOTE: This was copied from the existing custom-shape-func-that-raises.mlir
// and can be deleted once we fully move to using emit-mojo.

// CHECK-LABEL: --- Running 'custom_success':
// CHECK-NEXT: Hello
// CHECK-NEXT: 'custom_success' returned tensor<2xsi32> [1, 1]
mo.graph @custom_success<D0, D1 -> D2>(
  %arg0: !mo.tensor<[D0], f32>,
  %arg1: !mo.tensor<[D1], f32>
) -> !mo.tensor<[D2], si32> {

  mo.invoke_shape_func["custom_op_that_raises"]<() -> D3>(%arg0) : (!mo.tensor<[D0], f32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D3], si32>
  mo.output<D3> %val : !mo.tensor<[D3], si32>
}

// CHECK-LABEL: --- Running 'custom_raises':
// CHECK-NEXT: Hello
// CHECK: 'custom_raises' returned <error:
// CHECK: data.get_shape()[0] == 20
mo.graph @custom_raises<D0, D1 -> D2>(
  %arg0: !mo.tensor<[D0], f32>,
  %arg1: !mo.tensor<[D1], f32>
) -> !mo.tensor<[D2], si32> {

  mo.invoke_shape_func["custom_op_that_raises"]<() -> D3>(%arg1) : (!mo.tensor<[D1], f32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D3], si32>
  mo.output<D3> %val : !mo.tensor<[D3], si32>
}
