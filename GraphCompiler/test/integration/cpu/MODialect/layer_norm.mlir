// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// TODO(GEX-1573): need shape function.

mo.graph @layer_norm<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<3x2xf32>> : !mo.tensor<[3, 2], f32>
  %gamma = mo.constant <#M.dense_array<1.1, 2.1> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %beta  = mo.constant <#M.dense_array<-1.0, -2.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %eps   = mo.constant <#M.dense_array<0.00001> : tensor<f32>> : !mo.tensor<[], f32>
  %res   = mo.reduce.layer_norm<() -> D2, D3>(%input, %gamma, %beta, %eps) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], f32>
}
// CHECK: 'layer_norm' returned tensor<3x2xf32> [-2.0999{{[0-9]*}}e+00, 9.9957{{[0-9]*}}e-02, -2.0999{{[0-9]*}}e+00, 9.9957{{[0-9]*}}e-02, -2.0999{{[0-9]*}}e+00, 9.9957{{[0-9]*}}e-02]
