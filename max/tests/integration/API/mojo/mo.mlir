mo.graph @add(%0: !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) attributes {argument_names = ["input"], result_names = ["output"]} {
  %1 = "mo.constant"() {value = #M.dense_array<3.0, 1.0, -6.0, 2.0, 5.0> : tensor<5xf32>} : () -> !mo.tensor<[5], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}
