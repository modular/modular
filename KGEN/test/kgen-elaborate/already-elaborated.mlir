// RUN: kgen-elaborate-opt %s -elaborate-generators="test-diagnostics=true" -verify-diagnostics

// expected-remark @below {{Generator has already been specialized}}
kgen.generator @foo() {
  kgen.call @bar() : () -> ()
  kgen.return
}

// expected-remark @below {{Generator has already been specialized}}
kgen.generator @bar() {
  kgen.return
}
