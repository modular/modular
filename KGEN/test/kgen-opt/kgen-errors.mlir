// RUN: kgen-opt -allow-unregistered-dialect %s -verify-diagnostics -split-input-file -o /dev/null

kgen.generator @test() {
  // expected-error @+1 {{invalid use of parameter with no declaration "p"}}
  "someop" () {
    attr = #kgen.param.decl.ref<"p"> : i1
  } : () -> ()
  kgen.return
}

// -----

kgen.generator @test<p1>() {
  // expected-note @-1 {{previous declaration here}}
  "someop" () { // expected-error {{redeclaration of parameter "p1"}}
    paramDecls = [#kgen.param.decl<"p1" : i4>]
  } : () -> ()
  kgen.return
}

// -----

kgen.generator @test<p1>() {
  // expected-error @+1 {{unknown attribute kind in paramDecls list 41 : i32}}
  "someop" () {
    paramDecls = [41 : i32]
  } : () -> ()
  kgen.return
}

// -----

kgen.generator @test<p1>() {
  // expected-error @+1 {{invalid ParamDeclAttr outside of paramDecls attribute}}
  "someop" () {
    notParamDecls = #kgen.param.decl<"p1" : i4>
  } : () -> ()
  kgen.return
}

// -----

kgen.generator @parameter_results<p1 -> r1: i4>() {
  // expected-error @+1 {{parameter #0 is named "r7" but should be "r1"}}
  kgen.return<r7: i4 = 7>
}