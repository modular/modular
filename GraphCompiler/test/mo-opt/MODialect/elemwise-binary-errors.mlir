// RUN: mo-opt %s --split-input-file --verify-diagnostics

// expected-note@+1 {{prior use here}}
mo.graph @add(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 5], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // expected-error@+1 {{use of value '%arg1' expects different type than prior uses}}
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @sub(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 10], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // expected-error@+1 {{use of value '%arg1' expects different type than prior uses}}
  %0 = mo.sub(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @mul(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 10], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // expected-error@+1 {{use of value '%arg1' expects different type than prior uses}}
  %0 = mo.mul(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @div(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 10], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // expected-error@+1 {{use of value '%arg1' expects different type than prior uses}}
  %0 = mo.div(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// -----

mo.graph @pow(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 10], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // expected-error@+1 {{op requires the same shape for all operands and results}}
  %0 = mo.pow(%arg0 : !mo.tensor<[2, 3], f32>, %arg1 : !mo.tensor<[2, 10], f32>) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// -----

// expected-error@+1 {{invalid dtype in MO::TensorType}}
mo.graph @signless(%arg0: !mo.tensor<[], i32>) -> (!mo.tensor<[], i32>) {
  mo.output %arg0 : !mo.tensor<[], i32>
}

// -----

mo.graph @equal(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 10], f32>) -> (!mo.tensor<[2, 3], bool>) {
  // expected-error@+1 {{'mo.equal' op requires the same shape for all operands and results}}
  %0 = mo.equal(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 10], f32>) -> !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// -----

mo.graph @greater(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // expected-error@+1 {{'mo.greater' op result #0 must be boolean element type, but got '!mo.tensor<[2, 3], f32>'}}
  %0 = mo.greater(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// -----

mo.graph @greaters(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 10], bool>) {
  // expected-error@+1 {{'mo.greater' op requires the same shape for all operands and results}}
  %0 = mo.greater(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 10], bool>
  mo.output %0 : !mo.tensor<[2, 10], bool>
}

// -----

mo.graph @greater_equal(%arg0: !mo.tensor<[2, 3], si32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], bool>) {
  // expected-error@+1 {{'mo.greater_equal' op requires the same element type for all operands}}
  %0 = mo.greater_equal(%arg0, %arg1) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// -----

mo.graph @greater_equal(%arg0: !mo.tensor<[2, 10], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], bool>) {
  // expected-error@+1 {{'mo.greater_equal' op requires the same shape for all operands and results}}
  %0 = mo.greater_equal(%arg0, %arg1) : (!mo.tensor<[2, 10], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}
