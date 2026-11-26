// RUN: mo-opt %s --split-input-file --verify-diagnostics

// `$` is reserved for the parameter names that result type inference creates.
// The `*"..."` form is how the textual IR spells a name that is not a bareword.

// expected-error @below {{parameter name '$expr1' uses the '$' prefix reserved for names created during result type inference}}
mo.graph @input_param<*"$expr1">(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  mo.output %x : !mo.tensor<[4], f32>
}

// -----

// expected-error @below {{parameter name '$out' uses the '$' prefix reserved for names created during result type inference}}
mo.graph @result_param<D0 -> *"$out">(%x: !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  mo.output<D0> %x : !mo.tensor<[D0], f32>
}

// -----

mo.graph @declared_in_body<D0>(%x: !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  // expected-error @below {{parameter name '$expr2' uses the '$' prefix reserved for names created during result type inference}}
  kgen.param.declare *"$expr2" = <add(D0, 1)>
  mo.output %x : !mo.tensor<[D0], f32>
}
