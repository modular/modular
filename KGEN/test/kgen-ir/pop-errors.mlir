// RUN: not kgen-opt %s -verify-diagnostics -split-input-file -o /dev/null

kgen.kernel @pop_constant() -> !meta.scalar<si64> {
  // expected-error @+1 {{unexpected error: 'pop.constant' op expected the type of the constant input value ('f32') to be compatible with the dtype of the return value ('si64').}}
  %0 = pop.constant(32.0 : f32) : !meta.scalar<si64>
  kgen.return %0 : !meta.scalar<si64>
}

// -----

kgen.kernel @pop_constant() -> !meta.scalar<f32> {
  // expected-error @below {{expected '<'}}
  %0 = pop.constant(32 : si32) : f32
  kgen.return %0 : !meta.scalar<f32>
}

// -----


// COM: The value 16777217 is constructed so that it cannot be represented as a
// single-precision floating point value.

kgen.kernel @pop_constan2t() -> !meta.scalar<f32> {
  // expected-error @+1 {{unexpected error: 'pop.constant' op expected the type of the constant input value ('i32') to be compatible with the dtype of the return value ('f32').}}
  %0 = pop.constant(16777217 : i32) : <f32>
  kgen.return %0 : !meta.scalar<f32>
}
