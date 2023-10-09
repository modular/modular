// RUN: kgen-opt %s -elaborate-generators='diag-all-failures=true' -verify-diagnostics

// expected-remark @below {{other failed instantiations}}
kgen.generator @one_will_fail() {
  kgen.param.fork f = <[1, 2]>
  // expected-remark @below {{constraint failed: one}}
  kgen.param.assert <eq(f, 1)>, "one"
  kgen.return
}

kgen.generator export @and_one_will_live() {
  kgen.call @one_will_fail() : () -> ()
  kgen.return
}
