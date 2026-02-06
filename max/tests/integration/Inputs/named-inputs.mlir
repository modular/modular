// Model with inputs named "a", "b", "c", "d", and "e".
mo.graph @named_inputs(
    %0: !mo.tensor<[5], si32>,
    %1: !mo.tensor<[5], si32>,
    %2: !mo.tensor<[5], si32>,
    %3: !mo.tensor<[5], si32>,
    %4: !mo.tensor<[5], si32>
) -> (!mo.tensor<[5], si32>) attributes {
    argument_names = ["a", "b", "c", "d", "e"],
    result_names = ["output"]
} {
    %5 = mo.add(%0, %1) : !mo.tensor<[5], si32>
    %6 = mo.add(%5, %2) : !mo.tensor<[5], si32>
    %7 = mo.add(%6, %3) : !mo.tensor<[5], si32>
    %8 = mo.add(%7, %4) : !mo.tensor<[5], si32>
    mo.output %8 : !mo.tensor<[5], si32>
}
