// Model with dynamic input and output shapes
mo.graph @add(%0: !mo.tensor<[?], si32>, %1: !mo.tensor<[?], si32>) -> (!mo.tensor<[?], si32>) attributes {argument_names = ["input", "input2"], result_names = ["output"]} {
  %2 = mo.add(%0, %1) : !mo.tensor<[?], si32>
  mo.output %2 : !mo.tensor<[?], si32>
}
