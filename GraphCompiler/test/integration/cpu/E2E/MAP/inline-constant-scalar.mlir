// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'scalar_constant_sub':
// CHECK: 'scalar_constant_sub' returned tensor<2x2xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
mo.graph @scalar_constant_sub(%x: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>) {
  %s = mo.constant <#M.dense_array<2.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[2, 2], f32>
  %r = mo.sub(%b, %x) : !mo.tensor<[2, 2], f32>
  mo.output %r : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: Running 'broadcast_scalar_constant':
// CHECK: 'broadcast_scalar_constant' returned tensor<2x3xf32> [7.500000e+00, 7.500000e+00, 7.500000e+00, 7.500000e+00, 7.500000e+00, 7.500000e+00]
mo.graph @broadcast_scalar_constant() -> (!mo.tensor<[2, 3], f32>) {
  %s = mo.constant <#M.dense_array<7.500000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[2, 3], f32>
  mo.output %b : !mo.tensor<[2, 3], f32>
}
