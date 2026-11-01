// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=3x1x2x2 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @linalg_band_part_rank_4<D0, D1, D2, D3>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D0, D1, D2, D3], f32> {
  %1 = mo.constant <#M.dense_array<-1> : tensor<si32>> : !mo.tensor<[], si32>
  %2 = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %3 = mo.constant <#M.dense_array<0> : tensor<ui8>> : !mo.tensor<[], bool>
  %4 = mo.linalg.band_part(%arg0, %1, %2, %3) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[D0, D1, D2, D3], f32>
  mo.output %4 : !mo.tensor<[D0, D1, D2, D3], f32>
}
// CHECK{LITERAL}: 'linalg_band_part_rank_4' returned tensor<3x1x2x2xf32> [1.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]

mo.graph @linalg_band_part_rank_4_exclude<D0, D1, D2, D3>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D0, D1, D2, D3], f32> {
  %1 = mo.constant <#M.dense_array<-1> : tensor<si32>> : !mo.tensor<[], si32>
  %2 = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %3 = mo.constant <#M.dense_array<1> : tensor<ui8>> : !mo.tensor<[], bool>
  %4 = mo.linalg.band_part(%arg0, %1, %2, %3) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[D0, D1, D2, D3], f32>
  mo.output %4 : !mo.tensor<[D0, D1, D2, D3], f32>
}
// CHECK{LITERAL}: 'linalg_band_part_rank_4_exclude' returned tensor<3x1x2x2xf32> [0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00]
