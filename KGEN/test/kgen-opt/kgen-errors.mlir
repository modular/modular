// RUN: kgen-opt -allow-unregistered-dialect %s -verify-diagnostics -split-input-file -o /dev/null

kgen.generator @test() {
  // expected-error @+1 {{invalid use of parameter with no declaration "p"}}
  "someop" () {
    attr = #kgen.param.decl.ref<"p" : i1>
  } : () -> ()
  kgen.return
}

// -----

kgen.generator @test<p1>() {
  // expected-note @-1 {{previous declaration here}}
  "someop" () { // expected-error {{redeclaration of parameter "p1"}}
    parameterDecls = [#kgen.param.decl<"p1" : i4>]
  } : () -> ()
  kgen.return
}

// -----

kgen.generator @test<p1>() {
  // expected-error @+1 {{unknown attribute kind in parameterDecls list 41 : i32}}
  "someop" () {
    parameterDecls = [41 : i32]
  } : () -> ()
  kgen.return
}

// -----

kgen.generator @test<p1>() {
  // expected-error @+1 {{invalid ParamDeclAttr outside of parameterDecls attribute}}
  "someop" () {
    notParameterDecls = #kgen.param.decl<"p1" : i4>
  } : () -> ()
  kgen.return
}

