// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=2x8 --input-shapes=1x1 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// CHECK: 'mogg_split' returned tensor<1x8xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
// CHECK-NEXT: , tensor<1x8xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]

// We pass in the output shape parameters of split to separate how we derived
// those parameters from the mogg e2e testing.
mo.graph @mogg_split<D0, D1, D2>(
  %input: !mo.tensor<[D0, 8], f32>,
  %dummy: !mo.tensor<[D1, D2], si64>
) -> (!mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>) {

  %res:2 = mogg.split(%input) : (!mo.tensor<[D0, 8], f32>
  ) -> (!mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>)

  mo.output %res#0, %res#1 : !mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>
}
