// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones -result-output-style=full %t.mlir | FileCheck %s

mo.graph @cumsum(%ones : !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) {
  %x = mo.constant <#M.dense_array<-1.0, 1.0, 2.0, -2.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %inp = mo.mul(%ones, %x) : !mo.tensor<[5], f32>
  %res = mo.cumsum(%inp) [axis = 0, exclusive = 0, reverse = 0] : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  mo.output %res : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'cumsum' returned tensor<5xf32> [-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 5.000000e+00]

mo.graph @cumsum_reverse(%ones : !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) {
  %x = mo.constant <#M.dense_array<-1.0, 1.0, 2.0, -2.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %inp = mo.mul(%ones, %x) : !mo.tensor<[5], f32>
  %res = mo.cumsum(%inp) [axis = 0, exclusive = 0, reverse = 1] : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  mo.output %res : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'cumsum_reverse' returned tensor<5xf32> [5.000000e+00, 6.000000e+00, 5.000000e+00, 3.000000e+00, 5.000000e+00]

mo.graph @cumsum_exclusive(%ones : !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) {
  %x = mo.constant <#M.dense_array<-1.0, 1.0, 2.0, -2.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %inp = mo.mul(%ones, %x) : !mo.tensor<[5], f32>
  %res = mo.cumsum(%inp) [axis = 0, exclusive = 1, reverse = 0] : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  mo.output %res : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'cumsum_exclusive' returned tensor<5xf32> [0.000000e+00, -1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00]
