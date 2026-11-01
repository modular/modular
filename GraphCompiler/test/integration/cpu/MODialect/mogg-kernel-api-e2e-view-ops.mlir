// RUN: mo-opt %s -o %t.mlir --gc-pipeline --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=2 --model-inputs=ones | FileCheck %s

// Kernel under new API should print the following:
// CHECK: 'broadcast_static' returned tensor<3x2xf32> [1.500000e+00, 2.500000e+00, 1.500000e+00, 2.500000e+00, 1.500000e+00, 2.500000e+00]
mo.graph @broadcast_static() -> (!mo.tensor<[3, 2], f32>) {
  %x = mo.constant <#M.dense_array<1.5, 2.5> : tensor<2xf32>> : !mo.tensor<[1, 2], f32>
  %0 = mo.static.broadcast_to(%x) : !mo.tensor<[1, 2], f32> -> !mo.tensor<[3, 2], f32>
  mo.output %0 : !mo.tensor<[3, 2], f32>
}

// CHECK: 'broadcast_dyn' returned tensor<2xf32> [4.000000e+00, 4.000000e+00]
mo.graph @broadcast_dyn<D>(%x: !mo.tensor<[D], f32>) -> !mo.tensor<[D], f32> {
  %cst = mo.constant <#M.dense_array<4.> : tensor<1xf32>> : !mo.tensor<[1], f32>
  %bcast = mo.static.broadcast_to(%cst) : !mo.tensor<[1], f32> -> !mo.tensor<[D], f32>
  mo.output %bcast  : !mo.tensor<[D], f32>
}
