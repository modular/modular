// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// Test fused RMSNorm + RoPE on GPU.
// Input shape [2, 4], weight [4], cos/sin [2, 4].
// With cos=1 and sin=0, the output is just the RMS-normalized input.
// RMS of row 0 = sqrt((1^2+2^2+3^2+4^2)/4 + 1e-5) = sqrt(7.5 + 1e-5) ≈ 2.7386
// normed row 0 ≈ [0.3651, 0.7303, 1.0954, 1.4606]
// With cos=1, sin=0: result = normed * 1 + rotated * 0 = normed.

mo.graph @rms_norm_rope_gpu() -> !mo.tensor<[2, 4], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<2x4xf32>> : !mo.tensor<[2, 4], f32, gpu:0>
  %weight = mo.constant <#M.dense_array<1.0, 1.0, 1.0, 1.0> : tensor<4xf32>> : !mo.tensor<[4], f32, gpu:0>
  %cos_vals = mo.constant <#M.dense_array<1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0> : tensor<2x4xf32>> : !mo.tensor<[2, 4], f32, gpu:0>
  %sin_vals = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0> : tensor<2x4xf32>> : !mo.tensor<[2, 4], f32, gpu:0>

  %result = mo.composite.rms_norm_rope(%input, %weight, %cos_vals, %sin_vals)
    {epsilon = #M.dense_array<1.0e-5> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[2, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
     !mo.tensor<[2, 4], f32, gpu:0>, !mo.tensor<[2, 4], f32, gpu:0>)
    -> !mo.tensor<[2, 4], f32, gpu:0>

  %ch = mo.chain.create()
  %out, %ch2 = mo.transfer[%ch] %result : !mo.tensor<[2, 4], f32, gpu:0> to <"cpu", 0>
  mo.output %out : !mo.tensor<[2, 4], f32>
}

// CHECK: 'rms_norm_rope_gpu' returned tensor<2x4xf32>
// CHECK-SAME: 3.65{{[0-9]*}}e-01
