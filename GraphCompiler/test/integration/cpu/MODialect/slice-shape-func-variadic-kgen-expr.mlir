// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --input-shapes=2 --input-shapes=100 %t.mlir --result-output-style=full | FileCheck %s

// This test make sure MOGG properly lowers KGEN expression with variadic
// inputs (`stop_param` gets inlined into the `mogg.routine` of the slice shape
// func call).
mo.graph @slice_shape_func_variadic_kgen_expr<D0 -> D1>(
  %dummy: !mo.tensor<[D0], f32>,
  %input: !mo.tensor<[100], f32>
) -> !mo.tensor<[D1], si32> {

  %start = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[1], si32>
  %stop_param = mosh.param.to_value: !mosh.ape = <[mul_no_wrap(D0, D0, D0)]>
  %stop = mo.shape.to_tensor(%stop_param) -> !mo.tensor<[1], si32>
  %step = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[1], si32>

  mo.invoke_shape_func["mo.slice"]<() -> D2>(%input, %start, %stop, %step)
    [dataDeptTensors = [1 : i32, 2 : i32, 3 : i32]] :
    (!mo.tensor<[100], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D2], si32>

  mo.output<D2> %val : !mo.tensor<[D2], si32>
}
// CHECK{LITERAL}: 'slice_shape_func_variadic_kgen_expr' returned tensor<8xsi32>
