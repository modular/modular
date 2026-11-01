// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @slice_ok<D0, D1 -> D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %start: !mo.tensor<[2], si64>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[2], si64>) -> !mo.tensor<[D2, D3], f32> {
  %0 = rmo.mo.slice<() -> D4, D5>(%arg0, %start, %stop, %step) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %0 : !mo.tensor<[D4, D5], f32>
}

// -----

mo.graph @slice_bad_start<D0, D1, D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %start: !mo.tensor<[D2, D3], f32>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.slice' op operand #1 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<[D2, D3], f32>'}}
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @slice_bad_step<D0, D1, D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %start: !mo.tensor<[2], si64>, %stop: !mo.tensor<[D2, D3], f32>, %step: !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.slice' op operand #2 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<[D2, D3], f32>'}}
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[D2, D3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @slice_bad_stop<D0, D1, D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %start: !mo.tensor<[2], si64>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.slice' op operand #3 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<[D2, D3], f32>'}}
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @slice_bad_result<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %start: !mo.tensor<[2], si64>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], f16> {
  // expected-error @+1 {{'rmo.mo.slice' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], f16>
  mo.output %0 : !mo.tensor<[D0, D1], f16>
}
