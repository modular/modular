// RUN: kgen %s -emit -func="kernel:%t.o" -verify-diagnostics

// expected-error @below {{unhandled floating point dtype: f16}}
// expected-note @below {{see current operation}}
kgen.func public @kernel(%a: !meta.scalar<f16>) -> !meta.scalar<f16> {
  kgen.return %a : !meta.scalar<f16>
}
