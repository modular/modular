// RUN: mo-opt %s --split-input-file --verify-diagnostics

// CHECK: mo.graph @softmax(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 4], f32> {
mo.graph @softmax(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 4], f32>) {
  // expected-error@+1 {{op failed to verify that all of {input, result} have same shape}}
  %0 = mo.reduce.softmax(%arg0) [axis = 1] : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 4], f32>
  mo.output %0 : !mo.tensor<[2, 4], f32>
}

// -----

mo.graph @softmax_dtype<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> (!mo.tensor<[D0, D1], f64>) {
  // expected-error@+1 {{op failed to verify that all of {input, result} have same dtype}}
  %0 = mo.reduce.softmax(%arg0) [axis = 1] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f64>
  mo.output %0 : !mo.tensor<[D0, D1], f64>
}

// -----

// CHECK: mo.graph @logsoftmax(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 4], f32> {
mo.graph @logsoftmax(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 4], f32>) {
  // expected-error@+1 {{op failed to verify that all of {input, result} have same shape}}
  %0 = mo.reduce.logsoftmax(%arg0) [axis = 1] : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 4], f32>
  mo.output %0 : !mo.tensor<[2, 4], f32>
}

// -----

mo.graph @logsoftmax_dtype<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> (!mo.tensor<[D0, D1], f64>) {
  // expected-error@+1 {{op failed to verify that all of {input, result} have same dtype}}
  %0 = mo.reduce.logsoftmax(%arg0) [axis = 1] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f64>
  mo.output %0 : !mo.tensor<[D0, D1], f64>
}
