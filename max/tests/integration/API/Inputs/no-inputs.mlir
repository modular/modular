// Model that takes in no inputs and returns a 1-d tensor.
mo.graph @no_inputs() -> (!mo.tensor<[5], si32>) attributes {argument_names = [], result_names = ["output"]} {
    %0 = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 2, 3, 4, 5> : tensor<5xsi32>} : () -> !mo.tensor<[5], si32>
    mo.output %0 : !mo.tensor<[5], si32>
}
