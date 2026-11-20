// RUN: mo-opt %s -split-input-file -verify-diagnostics

// The result dtype is free, but the shape must match the inferred one.
mo.graph @result_shape_mismatch(%a: !mo.tensor<[2, 3], bf16>, %b: !mo.tensor<[3, 4], bf16>) -> !mo.tensor<[2, 5], f32> {
  // expected-error @+2 {{inferred type(s) '!mo.tensor<[2, 4], bf16>' are incompatible with return type(s) of operation '!mo.tensor<[2, 5], f32>'}}
  // expected-error @+1 {{failed to infer returned types}}
  %r = rmo.matmul(%a, %b) : (!mo.tensor<[2, 3], bf16>, !mo.tensor<[3, 4], bf16>) -> !mo.tensor<[2, 5], f32>
  mo.output %r : !mo.tensor<[2, 5], f32>
}
