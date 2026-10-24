// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @and_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], bool>, %arg1: !mo.tensor<[D0, D1], bool>) -> !mo.tensor<[D0, D1], bool> {
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<[D0, D1], bool>, !mo.tensor<[D0, D1], bool>) -> !mo.tensor<[D0, D1], bool>
  mo.output %1 : !mo.tensor<[D0, D1], bool>
}

// -----

mo.graph @and_bad_operand<D0, D1>(%arg0: !mo.tensor<[D0, D1], bool>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool> {
  // expected-error @+1 {{'rmo.mo.and' op operand #1 must be boolean element type, but got '!mo.tensor<[D0, D1], f32>'}}
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<[D0, D1], bool>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool>
  mo.output %1 : !mo.tensor<[D0, D1], bool>
}

// -----

mo.graph @and_bad_result<D0, D1>(%arg0: !mo.tensor<[D0, D1], bool>, %arg1: !mo.tensor<[D0, D1], bool>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.and' op result #0 must be boolean element type, but got '!mo.tensor<[D0, D1], f32>'}}
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<[D0, D1], bool>, !mo.tensor<[D0, D1], bool>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @or_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], bool>, %arg1: !mo.tensor<[D0, D1], bool>) -> !mo.tensor<[D0, D1], bool> {
  %1 = rmo.mo.or(%arg0, %arg1) : (!mo.tensor<[D0, D1], bool>, !mo.tensor<[D0, D1], bool>) -> !mo.tensor<[D0, D1], bool>
  mo.output %1 : !mo.tensor<[D0, D1], bool>
}

// -----

mo.graph @xor_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], bool>, %arg1: !mo.tensor<[D0, D1], bool>) -> !mo.tensor<[D0, D1], bool> {
  %1 = rmo.mo.xor(%arg0, %arg1) : (!mo.tensor<[D0, D1], bool>, !mo.tensor<[D0, D1], bool>) -> !mo.tensor<[D0, D1], bool>
  mo.output %1 : !mo.tensor<[D0, D1], bool>
}
