mo.graph @add(%0: !mo.tensor<[5], f32>, %1: !mo.tensor<[5], si32>) -> (!mo.tensor<[5], si32>) attributes {argument_names = ["input0", "input1"], result_names = ["output"]} {
  // Add a 0-array to the %1 input tensor so that the returned tensor
  // value is not literally the same borrowed pointer as was passed in,
  // to avoid lifetime issues that currently effect only this contrived test.
  %2 = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0, 0> : tensor<5xsi32>} : () -> !mo.tensor<[5], si32>
  %3 = mo.add(%1, %2) : !mo.tensor<[5], si32>
  mo.output %3 : !mo.tensor<[5], si32>
}
