// Model that returns the same tensor twice.
mo.graph @no_inputs(%arg: !mo.tensor<[5], si32>) -> (!mo.tensor<[5], si32>, !mo.tensor<[5], si32>) attributes {argument_names = ["input"], result_names = ["output1", "output2"]} {
    %0 = mo.add(%arg, %arg) : !mo.tensor<[5], si32>
    mo.output %0, %0 : !mo.tensor<[5], si32>, !mo.tensor<[5], si32>
}
