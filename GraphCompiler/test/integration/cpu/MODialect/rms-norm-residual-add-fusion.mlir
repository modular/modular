// RUN: mo-opt %s --gc-pipeline=constant-fold=false| FileCheck %s --check-prefix=CHECK-IR
// RUN: mt --execute --result-output-style=full %s | FileCheck %s --check-prefix=CHECK-MT

// Single-norm residual-add + RMSNorm boundary `rms_norm(input + residual)`
// (the canonical transformer/mamba pre-norm boundary) must fuse into
// `mo.composite.rms_norm_residual_add`, and the numerics must match the
// unfused add + rms_norm pair. The graph output consumes both the normalized
// result and the post-add intermediate so both fused outputs are exercised.

mo.graph @rms_norm_residual_add() -> !mo.tensor<[3, 2], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<3x2xf32>> : !mo.tensor<[3, 2], f32>
  %residual = mo.constant <#M.dense_array<0.5, 0.5, 0.5, 0.5, 0.5, 0.5> : tensor<3x2xf32>> : !mo.tensor<[3, 2], f32>
  %gamma = mo.constant <#M.dense_array<1.1, 2.1> : tensor<2xf32>> : !mo.tensor<[2], f32>

  // This should be fused into rms_norm_residual_add.
  %residual_add = mo.add(%input, %residual) : !mo.tensor<[3, 2], f32>
  %rnorm = mo.reduce.rms_norm(%residual_add, %gamma) {epsilon = #M.dense_array<0.00001> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[3, 2], f32>
  // Consume the intermediate (post-add) value too, so the fused op's second
  // output is used and not dead-code-eliminated.
  %out = mo.add(%rnorm, %residual_add) : !mo.tensor<[3, 2], f32>
  mo.output %out : !mo.tensor<[3, 2], f32>
}

// CHECK-MT: 'rms_norm_residual_add' returned tensor<3x2xf32> [2.3003{{[0-9]*}}e+00, 5.0466{{[0-9]*}}e+00, 4.4550{{[0-9]*}}e+00, 6.8442{{[0-9]*}}e+00, 6.5048{{[0-9]*}}e+00, 8.7671{{[0-9]*}}e+00]
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.composite.rms_norm_residual_add{{.*}}
// CHECK-NOT-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}"mo.reduce.rms_norm{{.*}}+mo.add{{.*}}
