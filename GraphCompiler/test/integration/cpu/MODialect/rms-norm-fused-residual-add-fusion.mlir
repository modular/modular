// RUN: mo-opt %s --gc-pipeline=constant-fold=false| FileCheck %s --check-prefix=CHECK-IR
// RUN: mt --execute --result-output-style=full %s | FileCheck %s --check-prefix=CHECK-MT

mo.graph @rms_norm_fused_residual_add() -> !mo.tensor<[3, 2], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<3x2xf32>> : !mo.tensor<[3, 2], f32>
  %gamma1 = mo.constant <#M.dense_array<1.1, 2.1> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %gamma2 = mo.constant <#M.dense_array<0.9, 1.9> : tensor<2xf32>> : !mo.tensor<[2], f32>

  // This should be fused into rms_norm_fused_residual_add
  %rnorm1 = mo.reduce.rms_norm(%input, %gamma1) {epsilon = #M.dense_array<0.00001> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[3, 2], f32>
  %residual_add = mo.add(%rnorm1, %input) : !mo.tensor<[3, 2], f32>
  %rnorm2 = mo.reduce.rms_norm(%residual_add, %gamma2) {epsilon = #M.dense_array<0.00002> : tensor<f32>, weight_offset = #M.dense_array<0.1> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[3, 2], f32>
  %downstream_residual_add = mo.add(%rnorm2, %residual_add) : !mo.tensor<[3, 2], f32>
  mo.output %downstream_residual_add : !mo.tensor<[3, 2], f32>
}

// CHECK-MT: 'rms_norm_fused_residual_add' returned tensor<3x2xf32> [2.1796{{[0-9]*}}e+00, 7.3139{{[0-9]*}}e+00, 4.6759{{[0-9]*}}e+00, 8.7830{{[0-9]*}}e+00, 6.8252{{[0-9]*}}e+00, 1.0572{{[0-9]*}}e+01]
// CHECK-NOT-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}"mo.reduce.rms_norm{{.*}}+mo.add{{.*}}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.composite.rms_norm_fused_residual_add{{.*}}
// CHECK-NOT-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}"mo.reduce.rms_norm{{.*}}+mo.add{{.*}}
