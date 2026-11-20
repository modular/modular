// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @equal_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool> {
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool>
  mo.output %1 : !mo.tensor<[D0, D1], bool>
}

// -----

mo.graph @equal_bad_operand<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], si32>) -> !mo.tensor<[D0, D1], bool> {
  // expected-error @+1 {{'rmo.mo.equal' op failed to verify that all of {input_x, input_y} have same dtype}}
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], si32>) -> !mo.tensor<[D0, D1], bool>
  mo.output %1 : !mo.tensor<[D0, D1], bool>
}

// -----

mo.graph @equal_bad_result<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.equal' op result #0 must be boolean element type, but got '!mo.tensor<[D0, D1], f32>'}}
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @greater_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool> {
  %1 = rmo.mo.greater(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool>
  mo.output %1 : !mo.tensor<[D0, D1], bool>
}

// -----

mo.graph @greater_equal_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool> {
  %1 = rmo.mo.greater_equal(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool>
  mo.output %1 : !mo.tensor<[D0, D1], bool>
}

// -----

mo.graph @not_equal_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool> {
  %1 = rmo.mo.not_equal(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool>
  mo.output %1 : !mo.tensor<[D0, D1], bool>
}
