// RUN: kgen-opt %s -elaborate-generators="test-diagnostics=true" -verify-diagnostics

// expected-remark @below {{Generator has already been specialized}}
kgen.generator @foo() {
  kgen.call @bar() : () -> ()
  kgen.return
}

// expected-remark @below {{Generator has already been specialized}}
kgen.generator @bar() {
  kgen.return
}

// No warnings or errors expected for this, it should just pass through with
// no changes.
kgen.func @alreadyDone() {
  kgen.return
}
