// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @pad_reflect_ok<D0, D1, D2 -> D3, D4, D5>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %pads: !mo.tensor<[6], si64>) -> !mo.tensor<[D3, D4, D5], f32> {
  %0 = rmo.mo.pad.reflect<() -> D6, D7, D8>(%arg0, %pads) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[6], si64>) -> !mo.tensor<[D6, D7, D8], f32>
  mo.output<D6, D7, D8> %0 : !mo.tensor<[D6, D7, D8], f32>
}

// -----

mo.graph @pad_reflect_params_ok<D0, D1, D2 -> D3, D4, D5>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %pads: !mo.tensor<[?], si64>) -> !mo.tensor<[D3, D4, D5], f32> {
  %0 = rmo.mo.pad.reflect<() -> D6, D7, D8>(%arg0, %pads) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[D6, D7, D8], f32>
  mo.output<D6, D7, D8> %0 : !mo.tensor<[D6, D7, D8], f32>
}

// -----

mo.graph @pad_reflect_static_ok<() -> D0, D1, D2>(%arg0: !mo.tensor<[2, 3, 4], f32>, %pads: !mo.tensor<[6], si64>) -> !mo.tensor<[D0, D1, D2], f32> {
  %0 = rmo.mo.pad.reflect<() -> D3, D4, D5>(%arg0, %pads) : (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[6], si64>) -> !mo.tensor<[D3, D4, D5], f32>
  mo.output<D3, D4, D5> %0 : !mo.tensor<[D3, D4, D5], f32>
}

// -----

mo.graph @pad_repeat_ok<D0, D1, D2 -> D3, D4, D5>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %pads: !mo.tensor<[6], si64>) -> !mo.tensor<[D3, D4, D5], f32> {
  %0 = rmo.mo.pad.repeat<() -> D6, D7, D8>(%arg0, %pads) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[6], si64>) -> !mo.tensor<[D6, D7, D8], f32>
  mo.output<D6, D7, D8> %0 : !mo.tensor<[D6, D7, D8], f32>
}

// -----

mo.graph @pad_constant_ok<D0, D1, D2 -> D3, D4, D5>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %pads: !mo.tensor<[6], si64>, %cst: !mo.tensor<[], f32>) -> !mo.tensor<[D3, D4, D5], f32> {
  %0 = rmo.mo.pad.constant<() -> D6, D7, D8>(%arg0, %pads, %cst) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[6], si64>, !mo.tensor<[], f32>) -> !mo.tensor<[D6, D7, D8], f32>
  mo.output<D6, D7, D8> %0 : !mo.tensor<[D6, D7, D8], f32>
}

// -----

mo.graph @pad_reflect_bad_paddings<D0, D1, D2, D3, D4>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %pads: !mo.tensor<[D3, D4], f32>) -> !mo.tensor<[D0, D1, D2], f32> {
  // expected-error @+1 {{'rmo.mo.pad.reflect' op operand #1 must be unranked or shape [N] and signed 64 or signed 32 integer elements, but got '!mo.tensor<[D3, D4], f32>'}}
  %0 = rmo.mo.pad.reflect(%arg0, %pads) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4], f32>) -> !mo.tensor<[D0, D1, D2], f32>
  mo.output %0 : !mo.tensor<[D0, D1, D2], f32>
}

// -----

mo.graph @pad_reflect_bad_result<D0, D1, D2>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %pads: !mo.tensor<[6], si64>) -> !mo.tensor<[D0, D1, D2], f16> {
  // expected-error @+1 {{'rmo.mo.pad.reflect' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.pad.reflect(%arg0, %pads) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[6], si64>) -> !mo.tensor<[D0, D1, D2], f16>
  mo.output %0 : !mo.tensor<[D0, D1, D2], f16>
}
