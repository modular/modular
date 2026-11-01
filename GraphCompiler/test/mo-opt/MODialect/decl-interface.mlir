// RUN: mo-opt --verify-parameters --verify-diagnostics

mo.graph @model(%cond : !mo.tensor<[], bool>) {
  mo.if(%cond : !mo.tensor<[], bool>) {
    mo.custom<() -> N> [symbol = "foo"]() : () -> ()
    mo.yield
  } else {
    mo.custom<() -> M> [symbol = "bar"]() : () -> ()
    mo.yield
  }
  // expected-error @below {{'mo.custom' op invalid use of parameter with no declaration "M"}}
  %0 = mo.custom [symbol = "baz"]() : () -> (!mo.tensor<[M, N], f32>)
  mo.output
}
