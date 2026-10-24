// RUN: mo-opt %s --gc-pipeline=constant-fold=false | FileCheck %s --check-prefix=CHECK-IR
// RUN: mt --execute --result-output-style=full %s | FileCheck %s --check-prefix=CHECK-MT


mo.graph @rms_norm() -> !mo.tensor<[3, 2], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<3x2xf32>> : !mo.tensor<[3, 2], f32>
  %gamma = mo.constant <#M.dense_array<1.1, 2.1> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %rnorm = mo.reduce.rms_norm(%input, %gamma) {epsilon = #M.dense_array<0.00001> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[3, 2], f32>
  %res   = mo.add(%rnorm, %input) : !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// CHECK-MT: 'rms_norm' returned tensor<3x2xf32> [1.6957{{[0-9]*}}e+00, 4.6563{{[0-9]*}}e+00, 3.9333{{[0-9]*}}e+00, 6.3758{{[0-9]*}}e+00, 5.9958{{[0-9]*}}e+00, 8.2815{{[0-9]*}}e+00]
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.rms_norm{{.*}}mo.add{{.*}}
