// RUN: mo-opt --gc-pipeline --split-input-file --verify-diagnostics

mo.graph @constant_unsupported_f32() -> !mo.scalar<f32> {
  // expected-error@+1 {{'mo.constant.scalar' op Only si32 and i1 constant scalar are currently lowerable}}
  %0 = mo.constant.scalar <1.0 : f32> : !mo.scalar<f32>
  mo.output %0 : !mo.scalar<f32>
}

// -----

mo.graph @constant_unsupported_si64() -> !mo.scalar<si64> {
  // expected-error@+1 {{'mo.constant.scalar' op Only si32 and i1 constant scalar are currently lowerable}}
  %0 = mo.constant.scalar <1 : si64> : !mo.scalar<si64>
  mo.output %0 : !mo.scalar<si64>
}

// -----

mo.graph @constant_unsupported_f64() -> !mo.scalar<f64> {
  // expected-error@+1 {{'mo.constant.scalar' op Only si32 and i1 constant scalar are currently lowerable}}
  %0 = mo.constant.scalar <1.0 : f64> : !mo.scalar<f64>
  mo.output %0 : !mo.scalar<f64>
}
