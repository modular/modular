// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @select_cond_vector(%cond: !mo.tensor<[2], bool>, %x: !mo.tensor<[2, 3], f32>, %y: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // expected-error@below {{'mo.select' op requires the same shape for all operands and results}}
  %0 = mo.select(%cond, %x, %y) : (!mo.tensor<[2], bool>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// -----

mo.graph @select_cond_mismatched_operands(%cond: !mo.tensor<[2, 3], bool>, %x: !mo.tensor<[2, 3], f32>, %y: !mo.tensor<[2], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // expected-error@below {{'mo.select' op requires the same shape for all operands and results}}
  %0 = mo.select(%cond, %x, %y) : (!mo.tensor<[2, 3], bool>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}
// -----

mo.graph @select_cond_mismatched_result(%cond: !mo.tensor<[2, 3], bool>, %x: !mo.tensor<[2, 3], f32>, %y: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2], f32>) {
  // expected-error@below {{'mo.select' op requires the same shape for all operands and results}}
  %0 = mo.select(%cond, %x, %y) : (!mo.tensor<[2, 3], bool>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2], f32>
  mo.output %0 : !mo.tensor<[2], f32>
}
