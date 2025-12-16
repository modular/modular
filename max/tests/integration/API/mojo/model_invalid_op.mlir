mo.graph @elaboration_error_in_kernel() -> !mo.tensor<[1], si32> {
  %0 = mo.custom {symbol = "fails_to_elaborate"}() : () -> (!mo.tensor<[1], si32>)
  mo.output %0 : !mo.tensor<[1], si32>
}
