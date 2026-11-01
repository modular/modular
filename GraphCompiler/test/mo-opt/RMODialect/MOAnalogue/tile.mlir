// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @tile_ok<D0, D1 -> D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %reps: !mo.tensor<[2], si64>) -> !mo.tensor<[D2, D3], f32> {
  %1 = rmo.mo.tile<() -> D4, D5>(%arg0, %reps) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %1 : !mo.tensor<[D4, D5], f32>
}

// -----

mo.graph @tile_bad_reps<D0, D1, D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %reps: !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error@below {{'rmo.mo.tile' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<[D2, D3], f32>'}}
  %1 = rmo.mo.tile(%arg0, %reps) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @tile_bad_result<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %reps: !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], f16> {
  // expected-error@below {{'rmo.mo.tile' op failed to verify that all of {input, result} have same dtype}}
  %1 = rmo.mo.tile(%arg0, %reps) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], f16>
  mo.output %1 : !mo.tensor<[D0, D1], f16>
}
