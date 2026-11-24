// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @select_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], bool>, %arg1: !mo.tensor<[D0, D1], f32>, %arg2: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<[D0, D1], bool>, !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @select_bad_condition<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>, %arg2: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error@below {{'rmo.mo.select' op operand #0 must be boolean element type, but got '!mo.tensor<[D0, D1], f32>'}}
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @select_bad_result<D0, D1>(%arg0: !mo.tensor<[D0, D1], bool>, %arg1: !mo.tensor<[D0, D1], f32>, %arg2: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f16> {
  // expected-error@below {{'rmo.mo.select' op failed to verify that all of {x, y, result} have same dtype}}
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<[D0, D1], bool>, !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f16>
  mo.output %1 : !mo.tensor<[D0, D1], f16>
}
