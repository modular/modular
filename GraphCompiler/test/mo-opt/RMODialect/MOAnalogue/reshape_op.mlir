// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @reshape_ok<D0, D1 -> D2, D3, D4>(%arg0: !mo.tensor<[D0, D1], f32>, %shape: !mo.tensor<[3], si64>) -> !mo.tensor<[D2, D3, D4], f32> {
  %1 = rmo.mo.reshape<() -> D5, D6, D7>(%arg0, %shape) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D5, D6, D7], f32>
  mo.output<D5, D6, D7> %1 : !mo.tensor<[D5, D6, D7], f32>
}

// -----

mo.graph @reshape_not_ok<D0, D1, D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %shape: !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error@below {{'rmo.mo.reshape' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<[D2, D3], f32>'}}
  %1 = rmo.mo.reshape(%arg0, %shape) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
