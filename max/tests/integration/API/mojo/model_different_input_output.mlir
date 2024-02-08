mo.graph @add(%0: !mo.tensor<[5], f32>, %1: !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) no_inline attributes {argument_names = ["input0", "input1"], result_names = ["output"]} {
  %2 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}
