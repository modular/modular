// RUN: kgen-opt -lower-lit -verify-parameters %s -verify-diagnostics

lit.struct.decl @A<b, c> { // expected-note {{@A declared here}}
}

// expected-error @+1 {{!kgen.declref symbol use input parameter #1 has name "e" but @A expected name "c"}}
lit.func @bad_litdeclref0(%x: !kgen.declref<@A<b = 10, e = 11>>) -> !lit.none {
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  kgen.return %0 : !lit.none
}
