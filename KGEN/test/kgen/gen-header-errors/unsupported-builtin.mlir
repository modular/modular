// RUN: kgen %s -emit -func="kernel" -o %t.o -verify-diagnostics

// expected-error @below {{unhandled float type: 'f128'}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func @kernel(%a: f128) -> f128 {
  kgen.return %a : f128
}

kgen.export [@kernel]
