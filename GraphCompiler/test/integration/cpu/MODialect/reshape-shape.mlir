// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

mo.graph @reshape_fallback_shape_func_smaller_rank<() -> D0, D1>(
  %input: !mo.tensor<[9, 2, 5], f32>
) -> !mo.tensor<[D0, D1], si32> {

  %target_shape = mo.constant <#M.dense_array<6, 15> : tensor<2xsi32>> : !mo.tensor<[2], si32>

  mo.invoke_shape_func["mo.reshape"]<() -> D2, D3>(%input, %target_shape)
    [dataDeptTensors = [1 : i32]] :
    (!mo.tensor<[9, 2, 5], f32>, !mo.tensor<[2], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D2, D3], si32>

  mo.output<D2, D3> %val : !mo.tensor<[D2, D3], si32>
}
// CHECK{LITERAL}: 'reshape_fallback_shape_func_smaller_rank' returned tensor<6x15xsi32>


mo.graph @reshape_fallback_shape_func_bigger_rank<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[9, 2, 5], f32>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  %target_shape = mo.constant <#M.dense_array<3, 5, 2, 3> : tensor<4xsi32>> : !mo.tensor<[4], si32>

  mo.invoke_shape_func["mo.reshape"]<() -> D4, D5, D6, D7>(%input, %target_shape)
    [dataDeptTensors = [1 : i32]] :
    (!mo.tensor<[9, 2, 5], f32>, !mo.tensor<[4], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
// CHECK{LITERAL}: 'reshape_fallback_shape_func_bigger_rank' returned tensor<3x5x2x3xsi32>


mo.graph @reshape_fallback_shape_func_inferred_dim<() -> D0, D1>(
  %input: !mo.tensor<[9, 2, 5], f32>
) -> !mo.tensor<[D0, D1], si32> {

  %target_shape = mo.constant <#M.dense_array<6, -1> : tensor<2xsi32>> : !mo.tensor<[2], si32>

  mo.invoke_shape_func["mo.reshape"]<() -> D2, D3>(%input, %target_shape)
    [dataDeptTensors = [1 : i32]] :
    (!mo.tensor<[9, 2, 5], f32>, !mo.tensor<[2], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D2, D3], si32>

  mo.output<D2, D3> %val : !mo.tensor<[D2, D3], si32>
}
// CHECK{LITERAL}: 'reshape_fallback_shape_func_inferred_dim' returned tensor<6x15xsi32>
