// RUN: mo-opt %s --split-input-file --verify-diagnostics

// expected-note@+1 {{prior use here}}
mo.graph @abs(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[4], f32>) {
  // expected-error@+1 {{use of value '%arg0' expects different type than prior uses}}
  %0 = mo.abs(%arg0) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @relu(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[4], f32>) {
  // expected-error@+1 {{use of value '%arg0' expects different type than prior uses}}
  %0 = mo.relu(%arg0) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @erf(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[4], f32>) {
  // expected-error@+1 {{use of value '%arg0' expects different type than prior uses}}
  %0 = mo.erf(%arg0) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @exp(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[4], f32>) {
  // expected-error@+1 {{use of value '%arg0' expects different type than prior uses}}
  %0 = mo.exp(%arg0) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @tanh(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[4], f32>) {
  // expected-error@+1 {{use of value '%arg0' expects different type than prior uses}}
  %0 = mo.tanh(%arg0) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// -----

mo.graph @floor(%arg0: !mo.tensor<[4, 5], si32>) -> (!mo.tensor<[4, 5], si32>) {
  // expected-error@+1 {{'mo.floor' op operand #0 must be floating point tensor, but got '!mo.tensor<[4, 5], si32>'}}
  %0 = mo.floor(%arg0) : !mo.tensor<[4, 5], si32>
  mo.output %0 : !mo.tensor<[4, 5], si32>
}

// -----

mo.graph @ceil(%arg0: !mo.tensor<[4, 5], si32>) -> (!mo.tensor<[4, 5], si32>) {
  // expected-error@+1 {{'mo.ceil' op operand #0 must be floating point tensor, but got '!mo.tensor<[4, 5], si32>'}}
  %0 = mo.ceil(%arg0) : !mo.tensor<[4, 5], si32>
  mo.output %0 : !mo.tensor<[4, 5], si32>
}

// -----

mo.graph @trunc(%arg0: !mo.tensor<[4, 5], si32>) -> (!mo.tensor<[4, 5], si32>) {
  // expected-error@+1 {{'mo.trunc' op operand #0 must be floating point tensor, but got '!mo.tensor<[4, 5], si32>'}}
  %0 = mo.trunc(%arg0) : !mo.tensor<[4, 5], si32>
  mo.output %0 : !mo.tensor<[4, 5], si32>
}

// -----

mo.graph @round(%arg0: !mo.tensor<[4, 5], si32>) -> (!mo.tensor<[4, 5], si32>) {
  // expected-error@+1 {{'mo.round' op operand #0 must be floating point tensor, but got '!mo.tensor<[4, 5], si32>'}}
  %0 = mo.round(%arg0) : !mo.tensor<[4, 5], si32>
  mo.output %0 : !mo.tensor<[4, 5], si32>
}

// -----

mo.graph @negative(%arg0: !mo.tensor<[4, 5], ui32>) -> (!mo.tensor<[4, 5], ui32>) {
  // expected-error@+1 {{'mo.negative' op operand #0 must be tensor with signed elements, but got '!mo.tensor<[4, 5], ui32>'}}
  %0 = mo.negative(%arg0) : !mo.tensor<[4, 5], ui32>
  mo.output %0 : !mo.tensor<[4, 5], ui32>
}

// -----

mo.graph @negative(%arg0: !mo.tensor<[4, 5], bool>) -> (!mo.tensor<[4, 5], bool>) {
  // expected-error@+1 {{'mo.negative' op operand #0 must be tensor with signed elements, but got '!mo.tensor<[4, 5], bool>'}}
  %0 = mo.negative(%arg0) : !mo.tensor<[4, 5], bool>
  mo.output %0 : !mo.tensor<[4, 5], bool>
}

// -----

mo.graph @sin(%arg0 : !mo.tensor<[4, 5], si64>) -> (!mo.tensor<[4, 5], si64>) {
  // expected-error@+1 {{'mo.sin' op operand #0 must be floating point tensor, but got '!mo.tensor<[4, 5], si64>'}}
  %0 = mo.sin(%arg0) : !mo.tensor<[4, 5], si64>
  mo.output %0 : !mo.tensor<[4, 5], si64>
}

// -----

mo.graph @cos(%arg0 : !mo.tensor<[4, 5], si64>) -> (!mo.tensor<[4, 5], si64>) {
  // expected-error@+1 {{'mo.cos' op operand #0 must be floating point tensor, but got '!mo.tensor<[4, 5], si64>'}}
  %0 = mo.cos(%arg0) : !mo.tensor<[4, 5], si64>
  mo.output %0 : !mo.tensor<[4, 5], si64>
}
