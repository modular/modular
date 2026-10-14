// RUN: kgen-opt %s -outline-closures -verify-diagnostics -split-input-file

kgen.generator @thin_nested_fn_captures() {
  // expected-note @below {{captured value defined here}}
  %0 = index.constant 0
  // expected-error @below {{nested function is marked as @noncapturing, but it captures values}}
  kgen.param.declare.region Fn = () -> index {
    // expected-note @below {{use of captured value here}}
    hlcf.return %0 : index
  }
  hlcf.return
}

// -----
kgen.generator @capturing_none() {
  // expected-note @below {{captured value defined here}}
  %none = kgen.param.constant: none = <#kgen.none>
  // expected-error @below {{we do not expect the capturing of None type.}}
  kgen.param.declare.region Fn = () capturing -> !kgen.none {
    hlcf.return %none : !kgen.none
  }
  hlcf.return
}
