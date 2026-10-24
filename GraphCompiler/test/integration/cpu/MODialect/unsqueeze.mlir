// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full --input-shapes=1 -model-inputs=negones %t.mlir | FileCheck %s

// indices_not_used is just there to work since `mt` invocation expects one input
mo.graph @calculate_unsqueeze_shape(%indices_not_used : !mo.tensor<[1], si32>) -> !mo.tensor<[7], si32> {
   %shape = mo.constant <#M.dense_array<10, 11, 12, 13> : tensor<4xsi32>> : !mo.tensor<[4], si32>
   %indices = mo.constant <#M.dense_array<0, -3, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
   %res = mo.unsqueeze_shape(%shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[7], si32>
   mo.output %res : !mo.tensor<[7], si32>
}
// CHECK: 'calculate_unsqueeze_shape' returned tensor<7xsi32> [1, 10, 11, 1, 1, 12, 13]

// note: %indices = [-1]
mo.graph @calculate_unsqueeze_shape_dynamic_indices(%indices : !mo.tensor<[1], si32>) -> !mo.tensor<[5], si32> {
   %shape = mo.constant <#M.dense_array<10, 11, 12, 13> : tensor<4xsi32>> : !mo.tensor<[4], si32>
   %res = mo.unsqueeze_shape(%shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[5], si32>
   mo.output %res : !mo.tensor<[5], si32>
}
// CHECK: 'calculate_unsqueeze_shape_dynamic_indices' returned tensor<5xsi32> [10, 11, 12, 13, 1]

mo.graph @calculate_unsqueeze_shape_dynamic_indices_with_reshape<() -> D0, D1, D2>(%indices : !mo.tensor<[1], si32>) -> !mo.tensor<[D0, D1, D2], si8> {
   %input_tensor = mo.constant <#M.dense_array<10, 11, 12, 13> : tensor<4xsi8>> : !mo.tensor<[2, 2], si8>
   %shape_tensor = mo.shape_of(%input_tensor) : (!mo.tensor<[2, 2], si8>) -> !mo.tensor<[2], si32>
   %new_shape_tensor = mo.unsqueeze_shape(%shape_tensor, %indices) : (!mo.tensor<[2], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[3], si32>
   %res = mo.reshape<() -> D3, D4, D5>(%input_tensor, %new_shape_tensor) : (!mo.tensor<[2, 2], si8>, !mo.tensor<[3], si32>) -> !mo.tensor<[D3, D4, D5], si8>
   mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], si8>
}
// CHECK: 'calculate_unsqueeze_shape_dynamic_indices_with_reshape' returned tensor<2x2x1xsi8>

mo.graph @unsqueeze_shape_fallback_shape_func<() -> D0>(
  %indices: !mo.tensor<[1], si32>
) -> !mo.tensor<[D0], si32> {

  %input = mo.constant <#M.dense_array<1, 2, 3, 4, 5> : tensor<5xsi32>> : !mo.tensor<[5], si32>

  mo.invoke_shape_func["mo.unsqueeze_shape"]<() -> D1>(%input, %indices)
    [dataDeptTensors = []] :
    (!mo.tensor<[5], si32>, !mo.tensor<[1], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D1], si32>

  mo.output<D1> %val : !mo.tensor<[D1], si32>
}
// CHECK: 'unsqueeze_shape_fallback_shape_func' returned tensor<6xsi32>
