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

// -----

// expected-note @+2 {{parameter defined with type 'ui32'}}
// expected-error @+1 {{reference to parameter "n" with incorrect type 'index'}}
kgen.generator @scalar_params_verbose<n : ui32>(%x : !meta.buffer<n, f32>) {
  kgen.return
}

// -----

// expected-error @+1 {{'undefined' does not reference a valid callee}}
kgen.call @undefined() : () -> ()

// -----

kgen.generator @g1(%x : i32) {
  // expected-error @+1 {{caller has 1 input but callee expects 0}}
  kgen.call @g2(%x) : (i32) -> ()
  kgen.return
}
kgen.generator @g2<>() { // expected-note {{callee declared here}}
  kgen.return
}

// -----

// expected-error @+1 {{'kgen.kernel' parameters not allowed in kgen.kernel, use kgen.generator instead}}
kgen.kernel @bad_kernel_param<>() {
  kgen.return
}

// -----

kgen.generator @g1(%x : i32) {
  // expected-error @+1 {{expected '('}}
  kgen.call @g2<()> : (i32) -> ()
  kgen.return
}
kgen.generator @g2<()>() {
  kgen.return
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @only_returns<p1 -> p2>() {
  kgen.return <p2 = p1>
}

kgen.kernel @test_only_returns() {
  // expected-error @+1 {{caller has 0 input parameters but callee expects 1}}
  kgen.call @only_returns<()->p2>() : () -> ()
  kgen.return
}

// -----

kgen.kernel @meta_buffer_size(%arg0: i32) -> index {
  // expected-error @+1 {{'meta.buffer.size' op operand #0 must be parameterized buffer type, but got 'i32'}}
  %0 = meta.buffer.size %arg0 : i32
  kgen.return %0 : index
}
// -----

kgen.kernel @bad_cast_size(%arg0 : !meta.buffer<42, f32>) -> !meta.buffer<1, f32> {
  // expected-error @+1 {{'meta.buffer.cast' op expected the size of the input buffer (42 : index) to be equal to the size you are casting it to (1 : index), or one of them to be unknown.}}
  %0 = meta.buffer.cast %arg0 : !meta.buffer<42, f32> to !meta.buffer<1, f32>
  kgen.return %0 : !meta.buffer<1, f32>
}

// -----
kgen.kernel @bad_cast_dtype(%arg0 : !meta.buffer<42, f32>) -> !meta.buffer<42, f64> {
  // expected-error @+1 {{'meta.buffer.cast' op expected the dtype of the input buffer (#kgen.dtype.constant<66 : i8> : !kgen.dtype) to the same as to the dtype you are casting to (#kgen.dtype.constant<67 : i8> : !kgen.dtype), or one of them to be unknown.}}
  %0 = meta.buffer.cast %arg0 : !meta.buffer<42, f32> to !meta.buffer<42, f64>
  kgen.return %0 : !meta.buffer<42, f64>
}

// -----

kgen.kernel @meta_buffer_dtype(%arg0: i32) -> !kgen.dtype {
  // expected-error @+1 {{'meta.buffer.dtype' op operand #0 must be parameterized buffer type, but got 'i32'}}
  %0 = meta.buffer.dtype %arg0 : i32
  kgen.return %0 : !kgen.dtype
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @only_returns<() -> p2: i4>() {
  kgen.return <p2: i4 = 2>
}

kgen.kernel @test_only_returns() {
  // expected-error @+1 {{caller output parameter #0 has type 'index' but caller expected type 'i4'}}
  kgen.call @only_returns<()->p2>() : () -> ()
  kgen.return
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @fn<p2>() {
  kgen.return
}

kgen.kernel @input_param_name() {
  // expected-error @+1 {{caller input parameter #0 has name "p1" but caller expected name "p2"}}
  kgen.call @fn<p1 = 42>() : () -> ()
  kgen.return
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @fn(%a: i1) -> i1 {
  kgen.return %a : i1
}

kgen.kernel @result_type(%a: i1) {
  // expected-error @+1 {{caller result #0 has type 'f32' but caller expected type 'i1'}}
  kgen.call @fn(%a) : (i1) -> f32
  kgen.return
}

// -----

kgen.generator.interface @itf<size>(si32) -> si32

// expected-error @+1 {{generator has parameters [#kgen.param.decl<size> : index, #kgen.param.decl<size2> : index] but interface @itf expects [#kgen.param.decl<size> : index]}}
kgen.generator @bad<size, size2>(%arg0: si32) -> si32
  implements @itf {
  kgen.return %arg0 : si32
}

// -----

kgen.generator.interface @itf<size>(si32, si32) -> si32

// expected-error @+1 {{generator has type (ui32, si32) -> si32 but interface @itf expects (si32, si32) -> si32}}
kgen.generator @bad<size>(%arg0: ui32, %arg1: si32) -> si32
  implements @itf {
  kgen.return %arg1 : si32
}

// -----

kgen.generator.interface @itf<size>(si32) -> si32

// expected-error @+1 {{generator has 0 input parameters, but interface @itf expects 1}}
kgen.generator @bad<() -> size>(%arg0: si32) -> si32
  implements @itf {
  kgen.return<size = 42> %arg0 : si32
}

// -----

// expected-error @+1 {{failed to satisfy constraint: any attribute valid parameter expression}}
%0 = kgen.param.value : i32 = <[]>
