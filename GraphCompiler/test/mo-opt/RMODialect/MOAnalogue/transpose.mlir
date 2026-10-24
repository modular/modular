// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @transpose_ok<D0, D1 -> D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %perm: !mo.tensor<[2], si64>) -> !mo.tensor<[D2, D3], f32> {
  %0 = rmo.mo.transpose<() -> D4, D5>(%arg0, %perm) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %0 : !mo.tensor<[D4, D5], f32>
}

// -----

mo.graph @transpose_bad_perm<D0, D1, D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %perm: !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.transpose' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<[D2, D3], f32>'}}
  %0 = rmo.mo.transpose(%arg0, %perm) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @transpose_bad_result<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %perm: !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], f16> {
  // expected-error @+1 {{'rmo.mo.transpose' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.transpose(%arg0, %perm) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], f16>
  mo.output %0 : !mo.tensor<[D0, D1], f16>
}
