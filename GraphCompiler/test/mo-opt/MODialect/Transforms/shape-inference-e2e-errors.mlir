// RUN: mo-opt %s --mo-shape-inference --split-input-file --verify-diagnostics

// TODO make first input dimension dynamic dynamic, which would require folding
// `eq(mul(D0, 10), mul(D0, 8))` in order to emit compile time error.
mo.graph @reshape_number_of_elems<() -> D2, D3, D4>(
  %input: !mo.tensor<[10, 10], f32>
) -> !mo.tensor<[D2, D3, D4], f32> {

  %axis = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %zero = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[1], si32>
  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[1], si32>

  %input_shape = mo.shape_of(%input) : (!mo.tensor<[10, 10], f32>) -> !mo.tensor<[2], si32>
  %d0 = mo.slice<() -> D0>(%input_shape, %zero, %one, %one) : (
    !mo.tensor<[2], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>
  ) -> !mo.tensor<[D0], si32>

  %dims = mo.constant <#M.dense_array<2, 4> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %target_shape = mo.concat<() -> D1>[0](%d0, %dims) : (
    !mo.tensor<[D0], si32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[D1], si32>

  // expected-error@below {{[reshape] input and output number of elements must match}}
  %result = mo.reshape<() -> D5, D6, D7>(%input, %target_shape) : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<[D1], si32>
  ) -> !mo.tensor<[D5, D6, D7], f32>

  mo.output<D5, D6, D7> %result : !mo.tensor<[D5, D6, D7], f32>
}

// -----

// This was adopted from an incorrect lowering of TF slice, which went silent
// without error. This is a minimized version of the repro test.
mo.graph @reshape_bad_annotated_rank<D0 -> D1, D2>(%arg0: !mo.tensor<[D0, 320], f32>) -> !mo.tensor<[D1, D2], f32> {
  %0 = mo.constant <#M.dense_array<0> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %1 = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %2 = mo.constant <#M.dense_array<1> : !M.array<1xsi32>> : !mo.tensor<[], si32>
  %4 = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %5 = mo.constant <#M.dense_array<1, 1> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %6 = mo.constant <#M.dense_array<0, 0> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %7 = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %8 = mo.slice<() -> D3, D4>(%arg0, %6, %7, %5) : (!mo.tensor<[D0, 320], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D3, D4], f32>
  %9 = mo.shape_of(%8) : (!mo.tensor<[D3, D4], f32>) -> !mo.tensor<[2], si64>
  %10 = mo.shape_of(%9) : (!mo.tensor<[2], si64>) -> !mo.tensor<[1], si32>
  %11 = mo.rebind<() -> D5>(%0) : !mo.tensor<[1], si32> -> !mo.tensor<[D5], si32>
  %12 = mo.static.reshape(%10) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %13 = mo.range<() -> D6>(%2, %12, %1) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D6], si32>
  %14 = mo.gather<() -> D7>(%9, %11) [axis = 0] : (!mo.tensor<[2], si64>, !mo.tensor<[D5], si32>) -> !mo.tensor<[D7], si64>
  %15 = mo.gather<() -> D8>(%9, %13) [axis = 0] : (!mo.tensor<[2], si64>, !mo.tensor<[D6], si32>) -> !mo.tensor<[D8], si64>
  %16 = mo.concat<() -> D9>[0] (%14, %4, %15) : (!mo.tensor<[D7], si64>, !mo.tensor<[1], si64>, !mo.tensor<[D8], si64>) -> !mo.tensor<[D9], si64>
  // expected-error@below {{shape function result #0 returned shape of rank 3, but the type annotation has shape of rank 2}}
  %17 = mo.reshape<() -> D10, D11>(%8, %16) : (!mo.tensor<[D3, D4], f32>, !mo.tensor<[D9], si64>) -> !mo.tensor<[D10, D11], f32>
  mo.output<D10, D11> %17 : !mo.tensor<[D10, D11], f32>
}

// -----

mo.graph @concat_axis_out_of_bound<D0, D1 -> D2, D3>(
  %input1: !mo.tensor<[D0, 1], f32>,
  %input2: !mo.tensor<[D1, 2], f32>
) -> !mo.tensor<[D2, D3], f32> {
  %axis = mo.constant <#M.dense_array<2> : tensor<si64>> : !mo.tensor<[], si64>
  // expected-error@below {{[concat] normalized axis must be within range [0, input_rank)}}
  %res = mo.concat<() -> D4, D5>[2](%input1, %input2) : (
    !mo.tensor<[D0, 1], f32>, !mo.tensor<[D1, 2], f32>
  ) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}
