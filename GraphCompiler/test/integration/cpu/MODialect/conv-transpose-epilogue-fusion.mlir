// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// An elementwise op on the conv_transpose output fuses into it as an
// epilogue, so the graph compiler selects the fused `execute` overload.
// With ones in, the transposed conv yields [1, 2, 1, 2, 4, 2, 1, 2, 1]; the
// epilogue adds 1.

mo.graph @conv_transpose_add_epilogue(
  %input: !mo.tensor<[1, 2, 2, 1], f32>,
  %filter: !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSFC>}>
) -> (!mo.tensor<[1, 3, 3, 1], f32>) {
  %st = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %op = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %conv = mo.conv_transpose(%input, %filter)
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op, filter_layout = ""] : (
    !mo.tensor<[1, 2, 2, 1], f32>, !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSFC>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[1, 3, 3, 1], f32>
  %one = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[1], f32>
  %bcast = mo.static.broadcast_to(%one) : !mo.tensor<[1], f32> -> !mo.tensor<[1, 3, 3, 1], f32>
  %add = mo.add(%conv, %bcast) : !mo.tensor<[1, 3, 3, 1], f32>
  mo.output %add : !mo.tensor<[1, 3, 3, 1], f32>
}
// CHECK{LITERAL}: 'conv_transpose_add_epilogue' returned tensor<1x3x3x1xf32> [2.000000e+00, 3.000000e+00, 2.000000e+00, 3.000000e+00, 5.000000e+00, 3.000000e+00, 2.000000e+00, 3.000000e+00, 2.000000e+00]
