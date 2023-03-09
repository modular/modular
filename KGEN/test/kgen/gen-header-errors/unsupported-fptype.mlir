// RUN: kgen %s -emit -o %t.o -verify-diagnostics

// expected-error @below {{bitwidth must be a power of 2}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func @kernel(%a: i24) -> i24 {
  kgen.return %a : i24
}

kgen.export @kernel to C
