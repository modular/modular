// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp --split-input-file --verify-diagnostics

kgen.func @foo() -> !kgen.scalar<si64> {
  %c = mosh.param.to_value = <42>
  hlcf.return %c : !kgen.scalar<si64>
}

mo.graph @unsupported_expr() {
  // expected-error@+1 {{cannot lower parameter expression}}
  kgen.param.declare fn: () -> si64 = <@foo>
  mo.output
}

// -----

mo.graph @unsupported_poc<X>(%arg0: !mo.tensor<[X], f32>) {
  // expected-error@+1 {{unsupported POC 'ceil_div_s'}}
  kgen.param.declare x: scalar<si64> = <ceil_div_s(:scalar<si64> X, 2)>
  mo.output
}
