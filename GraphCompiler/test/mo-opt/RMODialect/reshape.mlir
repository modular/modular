// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @reshape_neg_one(%input: !mo.tensor<[6], f32>) -> !mo.tensor<[2, 3], f32> {
  // expected-error@below {{[reshape] -1 size unsupported in target shape}}
  // expected-error@below {{'rmo.reshape' op failed to infer returned types}}
  %res = rmo.reshape(%input) [newShape = #mosh<ape[2, -1]> : !mosh.ape] : (!mo.tensor<[6], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}
