// RUN: kgen-opt -allow-unregistered-dialect %s -verify-diagnostics -split-input-file -o /dev/null

kgen.generator @test() {
  // expected-error @+1 {{invalid use of parameter with no declaration "p"}}
  "someop" () {
    attr = #kgen.param.decl.ref<p> : i1
  } : () -> ()
  kgen.return
}

// -----

kgen.generator @test<p1>() {
  // expected-note @-1 {{previous declaration here}}
  "someop" () { // expected-error {{redeclaration of parameter "p1"}}
    paramDecls = [#kgen.param.decl<p1> : i4]
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
    notParamDecls = #kgen.param.decl<p1> : i4
  } : () -> ()
  kgen.return
}

// -----

kgen.generator @parameter_results<p1 -> r1: i4>() {
  // expected-error @+1 {{parameter #0 is named "r7" but should be "r1"}}
  kgen.return<r7: i4 = 7>
}

// -----

"someop" () {
  use1 = #kgen.param.expr<add,
  // expected-error @+2 {{failed to parse ParamExprAttr parameter}}
  // expected-error @+1 {{parameter reference requires a type}}
                          #kgen.param.decl.ref<"p1">, 42 : si64>
} : () -> ()

// -----

"someop" () {
  // expected-error @+1 {{parameter declaration requires a type}}
  paramDecls = [#kgen.param.decl<"p3">]
} : () -> ()

// -----

"someop" () {
  // expected-error @+1 {{binary operators must have two operands}}
  use1 = #kgen.param.expr<shl, 1 : si32, 2 : si32, 3 : si32>
} : () -> ()

// -----

"someop" () {
  // expected-error @+1 {{operand type mismatch}}
  use1 = #kgen.param.expr<shl, 1 : si32, 2 : ui32>
} : () -> ()

// -----

// expected-error @+1 {{'kgen.param.value' binary operators must have two operands}}
%0 = kgen.param.value = <shl(p1, p2, p3)>

// -----

// expected-error @+1 {{'kgen.param.value' unknown expression invalid_op}}
%0 = kgen.param.value = <invalid_op(p1, p2, p3)>

// -----

// expected-error @+1 {{operator requires an index type}}
%0 = kgen.param.value : i32 = <shl(1, 2)>

// -----

// expected-error @+1 {{integer literal not valid for specified type}}
kgen.param.value : !kgen.dtype = <mul(1, 4)>

// -----

// expected-error @+1 {{kgen.dtype.constant requires i8 value}}
kgen.param.value : !kgen.dtype = <#kgen.dtype.constant<66 : i94>>

// -----

// expected-error @+1 {{kgen.dtype.constant requires !kgen.dtype type}}
kgen.param.value : i8 = <#kgen.dtype.constant<66 : i8>>

// -----

// expected-error @+1 {{kgen.dtype.constant requires !kgen.dtype type}}
kgen.param.value : i8 = <f32>

// -----

kgen.generator @scalar_params_verbose<n>(%x :
// expected-error @+1 {{expected '!kgen.dtype', but got 'index'}}
           !meta.scalar<#kgen.param.decl.ref<"n"> : index>) {
  kgen.return
}
  
// -----

// expected-error @+1 {{invalid use of parameter with no declaration "abc"}}
kgen.generator @scalar_params_verbose(%x : !meta.scalar<abc>) {
  kgen.return
}

// -----
kgen.generator @dtype_params() {
  // expected-error @+1 {{invalid use of parameter with no declaration "type"}}
  %y = "someop" () {} : () -> !meta.scalar<type>
  kgen.return
 }