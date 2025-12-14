mo.graph @append(%0: !mo.list<!mo.tensor<?,f32>>, %1: !mo.tensor<[5], f32>) -> (!mo.list<!mo.tensor<?, f32>>) attributes {argument_names = ["input_list", "input_tensor"], result_names = ["output_list"]} {
  %2 = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>} : () -> !mo.tensor<[3], f32>
  %3 = mo.list.append(%0, %2) : (
    !mo.list<!mo.tensor<?, f32>>, !mo.tensor<[3], f32>
  ) -> !mo.list<!mo.tensor<?, f32>>
  %4 = mo.list.append(%3, %1) : (
    !mo.list<!mo.tensor<?, f32>>, !mo.tensor<[5], f32>
  ) -> !mo.list<!mo.tensor<?, f32>>
  mo.output %4 : !mo.list<!mo.tensor<?, f32>>
}
