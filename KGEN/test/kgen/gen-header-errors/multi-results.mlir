// RUN: kgen %s -emit -o %t.o -verify-diagnostics

// expected-error @below {{functions with more than 1 result unsupported}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func public @kernel(%a: i32) -> (i32, i32) {
  kgen.return %a, %a : i32, i32
}
