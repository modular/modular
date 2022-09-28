// RUN: kgen-opt -allow-unregistered-dialect %s -verify-diagnostics -split-input-file -o /dev/null

kgen.generator @test() {
  // expected-error @+1 {{invalid use of parameter with no declaration "p"}}
  "someop" () {
    attr = #kgen.param.decl.ref<"p", i1>
  } : () -> ()
  kgen.return
}

// -----

kgen.generator @test<p1>() {
  // expected-note @-1 {{previous declaration here}}
  "someop" () { // expected-error {{redeclaration of parameter "p1"}}
    paramDecls = #kgen<param.decls[p1 : i4]>
  } : () -> ()
  kgen.return
}

// -----

"someop" () {
  // expected-error @+1 {{shl must have two operands}}
  use1 = #kgen.param.expr<shl, 1 : si32, 2 : si32, 3 : si32>
} : () -> ()

// -----

"someop" () {
  // expected-error @+1 {{operand type mismatch}}
  use1 = #kgen.param.expr<shl, 1 : si32, 2 : ui32>
} : () -> ()

// -----

// expected-error @+1 {{'kgen.param.constant' shl must have two operands}}
%0 = kgen.param.constant = <shl(p1, p2, p3)>

// -----

// expected-error @+1 {{'kgen.param.constant' unknown expression invalid_op}}
%0 = kgen.param.constant = <invalid_op(p1, p2, p3)>

// -----

// expected-error @+1 {{operator requires an index type}}
%0 = kgen.param.constant : i32 = <shl(1, 2)>

// -----

// expected-error @+1 {{integer literal not valid for specified type}}
kgen.param.constant : !kgen.dtype = <mul(1, 4)>

// -----

// expected-error @+1 {{kgen.dtype.constant requires !kgen.dtype type}}
kgen.param.constant : i8 = <#kgen.dtype.constant<f32>>

// -----

kgen.generator @foo() {
  // expected-error @+1 {{invalid use of parameter with no declaration "f32"}}
  kgen.param.constant : i8 = <f32>
  kgen.return
}

// -----

// expected-error @+2 {{reference to parameter "n" with incorrect type '!kgen.dtype'}}
//expected-note @+1 {{parameter defined with type 'index'}}
kgen.generator @scalar_params_verbose<n>(%x :
           !pop.scalar<#kgen.param.decl.ref<"n", index>>) {
  kgen.return
}

// -----

// expected-error @+1 {{invalid use of parameter with no declaration "abc"}}
kgen.generator @scalar_params_verbose(%x : !pop.scalar<abc>) {
  kgen.return
}

// -----

kgen.generator @dtype_params() {
  // expected-error @+1 {{invalid use of parameter with no declaration "type"}}
  %y = "someop" () {} : () -> !pop.scalar<type>
  kgen.return
}

// -----

// expected-error @below {{operator requires one operand}}
"someop"() {a = #kgen.param.expr<get_dtype, 1, 2>} : () -> ()

// -----

// expected-error @below {{operand should be a !kgen.mlirtype}}
"someop"() {a = #kgen.param.expr<get_dtype, 1>} : () -> ()

// -----

// expected-error @below {{should return a !kgen.dtype}}
"someop"() {a = #kgen.param.expr<get_dtype, #kgen.concretetype.constant<i32> : !kgen.mlirtype> : !kgen.mlirtype} : () -> ()

// -----

// expected-error @below {{does not implement DTypeInterface}}
"someop"() {a = #kgen.param.expr<get_dtype, #kgen.concretetype.constant<!foo<>>> : !kgen.dtype} : () -> ()

// -----

// expected-note @+2 {{parameter defined with type 'ui32'}}
// expected-error @+1 {{reference to parameter "n" with incorrect type 'index'}}
kgen.generator @scalar_params_verbose<n : ui32>(%x : !zap.buffer<n, f32>) {
  kgen.return
}

// -----

// expected-error @+1 {{'undefined' does not reference a valid callee}}
kgen.call @undefined() : () -> ()

// -----

kgen.generator @g1(%x : i32) {
  // expected-error @+1 {{caller has 1 argument but callee expects 0}}
  kgen.call @g2(%x) : (i32) -> ()
  kgen.return
}
kgen.generator @g2<>() { // expected-note {{callee declared here}}
  kgen.return
}

// -----

// expected-error @+1 {{kgen.func only allows output parameters, not input parameters}}
kgen.func @bad_param<x>() {
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

kgen.generator @only_returns<p1 -> index>() {
  kgen.return<p1>
}

kgen.func @test_only_returns() {
  // expected-error @+1 {{caller has 0 input parameters but callee expects 1}}
  kgen.call @only_returns<()->p2>() : () -> ()
  kgen.return
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @only_returns<() -> i4>() {
  kgen.return <:i4 2>
}

kgen.func @test_only_returns() {
  // expected-error @+1 {{caller result parameter #0 has type 'index' but callee expected type 'i4'}}
  kgen.call @only_returns<()->p2>() : () -> ()
  kgen.return
}

// -----

kgen.generator @fn<p2>() {
  kgen.return
}

kgen.func @input_param_name() {
  // expected-error @+1 {{caller input parameter #0 has name "p1" but callee expected name "p2"}}
  kgen.call @fn<p1 = 42>() : () -> ()
  kgen.return
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @fn(%a: i1) -> i1 {
  kgen.return %a : i1
}

kgen.func @result_type(%a: i1) {
  // expected-error @+1 {{caller result #0 has type 'f32' but callee expected type 'i1'}}
  kgen.call @fn(%a) : (i1) -> f32
  kgen.return
}

// -----

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf<size>(si32) -> si32

// expected-error @+1 {{generator has 2 input parameters but interface expects 1}}
kgen.generator @bad<size, size2>(%arg0: si32) -> si32
  implements @itf {
  kgen.return %arg0 : si32
}

// -----

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf<size>(si32, si32) -> si32

// expected-error @+1 {{generator argument #0 has type 'ui32' but interface expected type 'si32'}}
kgen.generator @bad<size>(%arg0: ui32, %arg1: si32) -> si32
  implements @itf {
  kgen.return %arg1 : si32
}

// -----

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf<size>(si32) -> si32

// expected-error @+1 {{generator has 0 input parameters but interface expects 1}}
kgen.generator @bad<() -> index>(%arg0: si32) -> si32 implements @itf {
  kgen.return<42> %arg0 : si32
}

// -----

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf<size>()

// expected-error @+1 {{generator input parameter #0 has name "barf" but interface expected name "size"}}
kgen.generator @bad<barf>() implements @itf {
  kgen.return
}

// -----

// expected-error @below {{expected attribute value}}
%0 = kgen.param.constant : i32 = <[:i32]>

// -----

kgen.generator.interface @take_and_return<p1 -> index>()

// expected-error @+1 {{invalid cyclic reference between operations defining and using parameters}}
kgen.func @self_cyclic() {
  // Uses r1 and defines r1
  kgen.call @take_and_return<p1 = r1 -> r1>() : () -> ()
  // expected-note @-1 {{this operation uses parameter "r1", which is defined by itself}}
  kgen.return
}

// -----

kgen.generator.interface @take_and_return<p1 -> index>()

// expected-error @+1 {{invalid cyclic reference between operations defining and using parameters}}
kgen.func @mutually_recursive() {
  // Uses r2 and defines r1
  kgen.call @take_and_return<p1 = r2 -> r1>() : () -> ()
  // expected-note @-1 {{this operation uses parameter "r2", which is defined by the first operation}}

  // Uses r1 and defines r2
  kgen.call @take_and_return<p1 = r1 -> r2>() : () -> ()
  // expected-note @-1 {{this operation uses parameter "r1", which is defined by:}}

  kgen.return
}

// -----

// expected-error @+1 {{invalid use of parameter with no declaration "he1ght"}}
kgen.generator @constrained<width, height>()
  constraints <[eq(width, 42), "thing"], [eq(he1ght, 42), "other"]> {
  kgen.return
}

// -----

// expected-error @+1 {{invalid use of parameter with no declaration "he1ght"}}
kgen.generator.interface @constrained<width, height>()
  constraints <[eq(width, 42), "width"], [eq(he1ght, 42), "height"]>

// -----

// expected-error @+1 {{invalid use of parameter with no declaration "ty2"}}
kgen.generator.interface @badTypes<ty1 : dtype>(%a : !pop.scalar<ty2>)

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @callee<type: dtype>(%x: !pop.scalar<type>) {
  kgen.return
}

kgen.generator @caller<type : dtype>(%arg0: !pop.scalar<type>) {
  // expected-error @+1 {{caller argument #0 has type '!pop.scalar<type>' but callee expected type '!pop.scalar<f64>'}}
  kgen.call @callee<type: dtype = f64>(%arg0) : (!pop.scalar<type>) -> ()
  kgen.return
}

// -----

// Reject uses of parameters in kgen.func since the elaborator should remove them.
kgen.func @test() {  // expected-note {{within kgen.func 'test'}}
  // Declaring parameters in a kernel is ok.
  kgen.param.declare someParam = <42>

  // Using them is not.
  // expected-error@+1 {{invalid use of parameter "someParam" in kgen.func}}
  "someop" () {} : () -> !pop.simd<someParam, f32>

  kgen.return
}

// -----

// kgen.func isn't allowed to call generators that take input parameters,
// but they are allowed to call generators with no input parameters.

kgen.generator @hasInputParam<param>() {
  kgen.return
}
kgen.generator @hasResultParam<() -> index>() {
  kgen.return<42>
}

kgen.func @test() {  // expected-note {{within kgen.func 'test'}}
  // ok
  kgen.call @hasResultParam<() -> result>() : () -> ()

  // expected-error@+1 {{cannot call generator with input arguments from concrete kgen.func}}
  kgen.call @hasInputParam<param = 42>() : () -> ()

  kgen.return
}

// -----

// expected-error @below {{expected type to be !kgen.mlirtype}}
"someop" () {value = #kgen.concretetype.constant<i32> : i32} : () -> ()

// -----

// expected-error @below {{expected type to be !kgen.mlirtype}}
"someop" () {value = #kgen.parameterizedtype.constant<i32> : i32} : () -> ()

// -----

// expected-error @below {{"dt" parameter not defined in signature}}
kgen.generator @region_params<r3: () -> !zap.buffer<4, dt>>() {
  kgen.return
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @takeUnary
  <unaryFn: <dt: dtype>(!pop.scalar<dt>) -> !pop.scalar<dt>>() {
  kgen.return
}

kgen.func @doubleExample(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  %0 = pop.add %arg0, %arg0: !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

kgen.generator @test_region() {
  // expected-error @+1 {{caller input parameter #0 has type}}
  kgen.call @takeUnary<
     unaryFn : (!pop.scalar<si32>) -> !pop.scalar<si32> = @doubleExample>() : () -> ()
  kgen.return
}

// -----

kgen.generator @takeFn<fn: () -> ()>() {
  kgen.return
}
kgen.generator @test() {
  // expected-error @+1 {{'@missing' does not reference a KGEN declaration}}
  kgen.call @takeFn<fn: ()->() = @missing>() : () -> ()
  kgen.return
}

// -----

kgen.generator @takeUnary
  <unaryFn: <dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt>>() {
  kgen.return
}

// expected-note @+1 {{@unary declared here}}
kgen.func @unary(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  kgen.return %arg0 : !pop.scalar<f32>
}

kgen.generator @test1() {
  // expected-error @+1 {{symbol use argument #0 has type '!pop.scalar<si32>' but @unary expected type '!pop.scalar<f32>'}}
  kgen.call @takeUnary<
     unaryFn : (!pop.scalar<si32>) -> !pop.scalar<si32> = @unary>() : () -> ()
  kgen.return
}

// -----

kgen.generator @takeUnary
  <unaryFn: <dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt>>() {
  kgen.return
}

// expected-note @+1 {{@unary2 declared here}}
kgen.generator @unary2<dt: dtype>(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  kgen.return %arg0 : !pop.scalar<si32>
}

kgen.generator @test2() {
  // expected-error @+1 {{symbol use has 0 input parameters but @unary2 expects 1}}
  kgen.call @takeUnary<
     unaryFn : (!pop.scalar<si32>) -> !pop.scalar<si32> = @unary2>() : () -> ()
  kgen.return
}

// -----

kgen.generator @call_param() {
  // expected-error @+1 {{'kgen.call_param' callee parameter type must be a signature type}}
  %0 = kgen.call_param[si32: 4]()
  kgen.return
}


// -----

kgen.generator @call_param<fn: <ty: type>()->()>() {
  // expected-error @+1 {{cannot name an operation with no results}}
  %0 = kgen.call_param[<ty: type>()->(): fn]<ty = 42>()
  kgen.return
}

// -----

kgen.generator @call_param<fn: <ty: type>()->()>() {
  // expected-error @+1 {{caller input parameter #0 has type 'index' but callee expected type}}
  kgen.call_param[<ty: type>()->(): fn]<ty = 42>()
  kgen.return
}

// -----

kgen.func @trivial(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

kgen.func @call_param_in_func(%arg0: si32) -> si32 {
  // expected-error @+1 {{kgen.call_param is only allowed in generators pre-elaboration}}
  %0 = kgen.call_param[(si32) -> si32: @trivial](%arg0)
  kgen.return %0: si32
}

// -----

kgen.generator @takeFn<unaryFn: <abc>()->()>() {
  kgen.return
}

// expected-note @+1 {{@thing declared here}}
kgen.generator @thing<dt>() {
  kgen.return
}

kgen.generator @test2() {
  // expected-error @+1 {{symbol use input parameter #0 has name "abc" but @thing expected name "dt"}}
  kgen.call @takeFn<unaryFn : <abc>()->() = @thing>() : () -> ()
  kgen.return
}

// -----

// expected-error @+1 {{"ty" parameter not defined in signature}}
kgen.generator @test<ty: type, p : <x>(!kgen.paramref<ty>)->()>() {
  kgen.return
}

// -----

// expected-error @+1 {{signature parameter "x" redefined}}
kgen.generator @test<ty: type, p : <x,x>()->()>
() {
  kgen.return
}

// -----

kgen.func @rebind(%a: !pop.scalar<f32>) {
  // expected-error @below {{cannot rebind concrete input type '!pop.scalar<f32>' to different concrete output type '!pop.scalar<si32>'}}
  %0 = kgen.rebind %a : !pop.scalar<f32> to !pop.scalar<si32>
  kgen.return
}

// -----

kgen.func @rebind(%a: !pop.scalar<f32>) {
  // expected-error @below {{cannot rebind concrete input type '!pop.scalar<f32>' to different concrete output type 'i32'}}
  %0 = kgen.rebind %a : !pop.scalar<f32> to i32
  kgen.return
}

// -----

kgen.generator @signature_taking_callee<fn: <size>() -> ()>() {
  kgen.return
}

kgen.generator @call_region() {
  // expected-note @below {{parameter declared here}}
  kgen.call @signature_taking_callee<fn: <size>() -> () = region>() : () -> ()
  // expected-error @below {{region has 1 argument but parameter expects 0}}
  fn<size>(%arg0: i32) {
    kgen.return
  }
  kgen.return
}

// -----

// expected-error @below {{unexpected result parameters}}
kgen.struct.decl @StructReturns<() -> dtype> {
}

// -----

// expected-error @below {{expected declaration body to have no arguments}}
"kgen.struct.decl"() ({
^bb0(%arg0: i32):
}) {sym_name = "StructArgs", constraints = #kgen<constraints[]>,
    paramDecls = #kgen<param.decls[]>, resultParamTypes = #kgen<type.array[]>} : () -> ()

// -----

// expected-error @below {{expected only `kgen.struct.field` ops in its body}}
"kgen.struct.decl"() ({
^bb0:
  // expected-note @below {{invalid child op here}}
  "not_struct_field"() : () -> ()
}) {sym_name = "StructArgs", constraints = #kgen<constraints[]>,
    paramDecls = #kgen<param.decls[]>, resultParamTypes = #kgen<type.array[]>} : () -> ()

// -----

kgen.struct.decl @StructDuplicate {
  // expected-note @below {{see previous declaration here}}
  x : i32
  y : i32
  // expected-error @below {{duplicate struct field "x"}}
  x : i32
}

// -----

kgen.struct.decl @SomeType<v, b> {}

// expected-error @below {{invalid use of parameter with no declaration "c"}}
kgen.generator.interface @InvalidTypeParamValue<a>() ->
    !kgen.typedef<@SomeType<v = a, b = c>>

// -----

// expected-note @below {{@SomeType declared here}}
kgen.struct.decl @SomeType<v, d> {}

// expected-error @below {{typedef symbol use input parameter #1 has name "b" but @SomeType expected name "d"}}
kgen.generator.interface @InvalidTypeParamValue<a, c>() ->
    !kgen.typedef<@SomeType<v = a, b = c>>
