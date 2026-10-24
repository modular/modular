// RUN: (mt --execute --result-output-style=full --input-shapes=1x2 --input-shapes=3 --input-shapes=2 --input-shapes="" --input-shapes=1x2 %s || true) | FileCheck %s --check-prefix=CHECK-WRONG-GAMMA
// RUN: (mt --execute --result-output-style=full --input-shapes=1x2 --input-shapes=2 --input-shapes=3 --input-shapes="" --input-shapes=1x2 %s || true) | FileCheck %s --check-prefix=CHECK-WRONG-BETA
// RUN: (mt --execute --result-output-style=full --input-shapes=1x2 --input-shapes=2 --input-shapes=2 --input-shapes="" --input-shapes=2x2 %s || true) | FileCheck %s --check-prefix=CHECK-WRONG-OUTPUT

// CHECK-WRONG-GAMMA: Gamma size does not match dimension of reduction
// CHECK-WRONG-BETA: Beta size does not match dimension of reduction
// CHECK-WRONG-OUTPUT: Input and output buffers are not same shape

mo.graph @layer_norm_wrong_gamma<A, B, C, D, E, F>(
  %input: !mo.tensor<[A, B], f32>,
  %gamma: !mo.tensor<[C], f32>,
  %beta: !mo.tensor<[D], f32>,
  %epsilon: !mo.tensor<[], f32>,
  %dummy_output_shape: !mo.tensor<[E, F], f32>
) -> !mo.tensor<[E, F], f32> {
  %res = mo.reduce.layer_norm(
    %input,
    %gamma,
    %beta,
    %epsilon
  ): (
    !mo.tensor<[A, B], f32>,
    !mo.tensor<[C], f32>,
    !mo.tensor<[D], f32>,
    !mo.tensor<[], f32>
  ) -> !mo.tensor<[E, F], f32>
  mo.output %res: !mo.tensor<[E, F], f32>
}
