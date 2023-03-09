// RUN: kgen %s -emit -o %t.o -verify-diagnostics -ignore-failure

// expected-error @below {{unhandled floating point dtype: f16}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func @kernel(%a: !pop.simd<1, f16>) -> !pop.simd<1, f16> {
  kgen.return %a : !pop.simd<1, f16>
}

kgen.export @kernel to C
