// RUN: kgen %s -emit -func="kernel:%t.o" -verify-diagnostics

// expected-error @below {{unhandled floating point dtype: f16}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func public @kernel(%a: !pop.scalar<f16>) -> !pop.scalar<f16> {
  kgen.return %a : !pop.scalar<f16>
}
