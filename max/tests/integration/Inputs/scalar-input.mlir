// Model that takes in a scalar input.
mo.graph @scalar_input(%0: !mo.tensor<[], si32>, %1: !mo.tensor<[5], si32>) -> (!mo.tensor<[5], si32>) attributes {argument_names = ["input0", "input1"], result_names = ["output"]} {
    %2 = mo.shape_of(%1) : (!mo.tensor<[5], si32>) -> !mo.tensor<[1], si64>
    %3 = mo.broadcast_to(%0, %2) : (!mo.tensor<[], si32>, !mo.tensor<[1], si64>) -> !mo.tensor<[5], si32>
    %4 = mo.add(%1, %3) : !mo.tensor<[5], si32>
    mo.output %4 : !mo.tensor<[5], si32>
}
