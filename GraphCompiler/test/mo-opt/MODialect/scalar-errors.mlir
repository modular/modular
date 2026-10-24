// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @scalar_type() {
  // expected-error@+1 {{mo.constant.scalar expected the dtype to be an integer}}
  %0 = mo.constant.scalar <1 : si64> : !mo.scalar<f32>
  mo.output
}

// -----

mo.graph @scalar_type() {
  // expected-error@+1 {{mo.constant.scalar expected the dtype to be a float}}
  %0 = mo.constant.scalar <2.5 : f32> : !mo.scalar<si64>
  mo.output
}

// -----

mo.graph @scalar_type() {
  // expected-error@+1 {{mo.constant.scalar expected bitwidth of the dtype and attribute to match}}
  %0 = mo.constant.scalar <1.0 : f64> : !mo.scalar<f32>
  mo.output
}

// -----

mo.graph @scalar_type() {
  // expected-error@+1 {{mo.constant.scalar expected bitwidth of the dtype and attribute to match}}
  %0 = mo.constant.scalar <1 : i8> : !mo.scalar<bool>
  mo.output
}
