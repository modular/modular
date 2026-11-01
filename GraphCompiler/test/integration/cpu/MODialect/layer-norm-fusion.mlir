// RUN: mo-opt %s --gc-pipeline=constant-fold=false | FileCheck %s --check-prefix=CHECK-IR
// RUN: mt --execute --result-output-style=full %s | FileCheck %s --check-prefix=CHECK-MT


mo.graph @layer_norm() -> !mo.tensor<[3, 2], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<3x2xf32>> : !mo.tensor<[3, 2], f32>
  %gamma = mo.constant <#M.dense_array<1.1, 2.1> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %beta  = mo.constant <#M.dense_array<-1.0, -2.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %eps   = mo.constant <#M.dense_array<0.00001> : tensor<f32>> : !mo.tensor<[], f32>
  %lnorm = mo.reduce.layer_norm(%input, %gamma, %beta, %eps) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[3, 2], f32>
  %res   = mo.add(%lnorm, %input) : !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// CHECK-MT: 'layer_norm' returned tensor<3x2xf32> [-1.0999{{[0-9]*}}e+00, 2.0999{{[0-9]*}}e+00, 9.0002{{[0-9]*}}e-01, 4.0999{{[0-9]*}}e+00, 2.9000{{[0-9]*}}e+00, 6.0999{{[0-9]*}}e+00]
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.layer_norm{{.*}}mo.add{{.*}}
