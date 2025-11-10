// RUN: kgen %s -emit=header -verify-diagnostics -ignore-failure

// expected-error @below {{unhandled floating point dtype: f16}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func export C @kernel(%a: !pop.simd<1, f16>) -> !pop.simd<1, f16> {
  kgen.return %a : !pop.simd<1, f16>
}
