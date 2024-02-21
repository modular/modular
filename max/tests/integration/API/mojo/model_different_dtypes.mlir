mo.graph @add(%0: !mo.tensor<[5], f32>, %1: !mo.tensor<[5], si32>) -> (!mo.tensor<[5], si32>) no_inline attributes {argument_names = ["input0", "input1"], result_names = ["output"]} {
  mo.output %1 : !mo.tensor<[5], si32>
}
