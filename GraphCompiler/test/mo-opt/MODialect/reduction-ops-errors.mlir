// RUN: mo-opt %s --split-input-file --verify-diagnostics

// COM: These tests are specific to ops, but verify properties that apply to all reduction ops.

mo.graph @input_res_rank_mismatch(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?], f32> {
  // expected-error @below {{input and output must either both have dynamic ranks or both have the same size static ranks}}
  %output = mo.reduce.add(%input) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @axis_too_big(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], f32> {
  // expected-error @below {{'mo.reduce.mul' op axis must be between [-3, 2] found 3}}
  %output = mo.reduce.mul(%input) [axis = 3] : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?, ?], f32>
}

// -----

mo.graph @axis_too_small(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], si64> {
  // expected-error @below {{'mo.reduce.arg_max' op axis must be between [-3, 2] found -4}}
  %output = mo.reduce.arg_max(%input) [axis = -4] : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], si64>
  mo.output %output : !mo.tensor<[?, ?, ?], si64>
}

// -----

mo.graph @reduced_dim_not_one(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 2], f32> {
  // expected-error @below {{'mo.reduce.add' op must turn exactly one input dimension into 1}}
  %output = mo.reduce.add(%input) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 2], f32>
  mo.output %output : !mo.tensor<[?, ?, 2], f32>
}

// -----

mo.graph @reduced_dim_dynamic(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], f32> {
  // expected-error @below {{'mo.reduce.max' op must turn exactly one input dimension into 1}}
  %output = mo.reduce.max(%input) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?, ?], f32>
}

// -----

mo.graph @two_reduced_dims(%input: !mo.tensor<[4, 3, 5], f32>) -> !mo.tensor<[4, 1, 1], f32> {
  // expected-error @below {{'mo.reduce.min' op must turn exactly one input dimension into 1}}
  %output = mo.reduce.min(%input) : (!mo.tensor<[4, 3, 5], f32>) -> !mo.tensor<[4, 1, 1], f32>
  mo.output %output : !mo.tensor<[4, 1, 1], f32>
}

// -----

mo.graph @mixed_dtypes_float_to_int(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], si32> {
  // expected-error @below {{'mo.reduce.add' op failed to verify that all of {input, result} have same dtype}}
  %output = mo.reduce.add(%input) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], si32>
  mo.output %output : !mo.tensor<[?, ?, 1], si32>
}

// -----

mo.graph @mixed_dtypes_float_to_float(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f16> {
  // expected-error @below {{'mo.reduce.max' op failed to verify that all of {input, result} have same dtype}}
  %output = mo.reduce.max(%input) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f16>
  mo.output %output : !mo.tensor<[?, ?, 1], f16>
}
