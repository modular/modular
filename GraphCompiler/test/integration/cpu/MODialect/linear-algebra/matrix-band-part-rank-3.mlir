// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=1x3x3 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @linalg_band_part_rank_3<D0, D1, D2>(%arg0: !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D0, D1, D2], f32> {
  %1 = mo.constant <#M.dense_array<-1> : tensor<si32>> : !mo.tensor<[], si32>
  %2 = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %3 = mo.constant <#M.dense_array<0> : tensor<ui8>> : !mo.tensor<[], bool>
  %4 = mo.linalg.band_part(%arg0, %1, %2, %3) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[D0, D1, D2], f32>
  mo.output %4 : !mo.tensor<[D0, D1, D2], f32>
}
// CHECK{LITERAL}: 'linalg_band_part_rank_3' returned tensor<1x3x3xf32> [1.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]

mo.graph @linalg_band_part_rank_3_exclude<D0, D1, D2>(%arg0: !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D0, D1, D2], f32> {
  %1 = mo.constant <#M.dense_array<-1> : tensor<si32>> : !mo.tensor<[], si32>
  %2 = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %3 = mo.constant <#M.dense_array<1> : tensor<ui8>> : !mo.tensor<[], bool>
  %4 = mo.linalg.band_part(%arg0, %1, %2, %3) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[D0, D1, D2], f32>
  mo.output %4 : !mo.tensor<[D0, D1, D2], f32>
}
// CHECK{LITERAL}: 'linalg_band_part_rank_3_exclude' returned tensor<1x3x3xf32> [0.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
