// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

mo.graph @tile() -> !mo.tensor<[6, 6], si32> {
  %input = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5> : tensor<6xsi32>> : !mo.tensor<[2, 3], si32>
  %repeats = mo.constant <#M.dense_array<3, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %output = mo.tile(%input, %repeats) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[6, 6], si32>
  mo.output %output : !mo.tensor<[6, 6], si32>
}
// CHECK{LITERAL}: 'tile' returned tensor<6x6xsi32> [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5]

mo.graph @tile_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[2, 3, 7, 11], f32>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  %repeats = mo.constant <#M.dense_array<2, 3, 2, 4> : tensor<4xsi64>> : !mo.tensor<[4], si64>

  mo.invoke_shape_func["mo.tile"]<() -> D4, D5, D6, D7>(%input, %repeats)
    [dataDeptTensors = [1 : i32]] :
    (!mo.tensor<[2, 3, 7, 11], f32>, !mo.tensor<[4], si64>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
// CHECK{LITERAL}: 'tile_fallback_shape_func' returned tensor<4x9x14x44xsi32>
