// RUN: kgen %s -emit=header -func="kernel" -verify-diagnostics

// expected-error @below {{unhandled elementary type: 'f128'}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func export C @kernel(%a: f128) -> f128 {
  kgen.return %a : f128
}
