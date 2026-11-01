// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false --allow-unregistered-dialect -o %t.mlir
// RUN: mt --execute --result-output-style=full --input-shapes=1 -model-inputs=negones %t.mlir | FileCheck %s

mo.graph @calculate_squeeze_shape_dynamic_indices_with_reshape<() -> D0, D1, D2>(%indices : !mo.tensor<[1], si32>) -> !mo.tensor<[D0, D1, D2], si8> {
   %input_tensor = mo.constant <#M.dense_array<10, 11, 12, 13> : tensor<4xsi8>> : !mo.tensor<[2, 1, 2, 1], si8>
   %shape_tensor = mo.shape_of(%input_tensor) : (!mo.tensor<[2, 1, 2, 1], si8>) -> !mo.tensor<[4], si32>
   %new_shape_tensor = mo.squeeze_shape(%shape_tensor, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[3], si32>
   %res = mo.reshape<() -> D3, D4, D5>(%input_tensor, %new_shape_tensor) : (!mo.tensor<[2, 1, 2, 1], si8>, !mo.tensor<[3], si32>) -> !mo.tensor<[D3, D4, D5], si8>
   mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], si8>
}
// CHECK: 'calculate_squeeze_shape_dynamic_indices_with_reshape' returned tensor<2x1x2xsi8>
