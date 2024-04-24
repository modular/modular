// RUN: kgen-opt -verify-parameters -lower-lit -split-input-file -verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// -----
// expected-error @-2 {{cyclic dependencies between global variables in 'lower-lit' pass}}

lit.globalvar.decl @foo : index {
  lit.globalvar.ref @bar : <index, mut #lit.lifetime>
}, {
}

lit.globalvar.decl @bar : index {
  lit.globalvar.ref @foo : <index, mut #lit.lifetime>
}, {
}
