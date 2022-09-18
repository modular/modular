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
    paramDecls = #kgen<param.decls[p1 : i4]>
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
  // expected-error @+2 {{failed to parse ParamOperatorAttr parameter}}
  // expected-error @+1 {{parameter reference requires a type}}
                          #kgen.param.decl.ref<"p1">, 42 : si64>
} : () -> ()

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

kgen.generator @scalar_params_verbose<n>(%x :
// expected-error @+1 {{expected '!kgen.dtype', but got 'index'}}
           !meta.scalar<#kgen.param.decl.ref<n> : index>) {
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

// expected-note @+1 {{callee declared here}}
kgen.generator @only_returns<p1 -> p2>() {
  kgen.return <p2 = p1>
}

kgen.func @test_only_returns() {
  // expected-error @+1 {{caller has 0 input parameters but callee expects 1}}
  kgen.call @only_returns<()->p2>() : () -> ()
  kgen.return
}

// -----

kgen.func @bad_scalar_rebind_dtype(%arg0: !meta.scalar<f32>) -> !meta.scalar<f64> {
  // expected-error @below {{input scalar dtype 'f32' disagrees with result scalar dtype 'f64'}}
  %0 = meta.scalar.rebind %arg0 : !meta.scalar<f32> to !meta.scalar<f64>
  kgen.return %0 : !meta.scalar<f64>
}

// -----

kgen.func @bad_pointer_rebind_dtype(%arg0: !meta.pointer<!meta.scalar<f32>>) -> !meta.pointer<!meta.scalar<f64>> {
  // expected-error @below {{input pointer element type '!meta.scalar<f32>' disagrees with result pointer element type '!meta.scalar<f64>'}}
  %0 = meta.pointer.rebind %arg0 : !meta.pointer<!meta.scalar<f32>> to !meta.pointer<!meta.scalar<f64>>
  kgen.return %0 : !meta.pointer<!meta.scalar<f64>>
}

// -----

kgen.func @bad_simd_rebind_size(%arg0 : !meta.simd<2, f32>) -> !meta.simd<1, f32> {
  // expected-error @+1 {{input SIMD size '2' disagrees with result SIMD size '1'}}
  %0 = meta.simd.rebind %arg0 : !meta.simd<2, f32> to !meta.simd<1, f32>
  kgen.return %0 : !meta.simd<1, f32>
}

// -----

kgen.func @bad_simd_rebind_dtype(%arg0 : !meta.simd<2, f32>) -> !meta.simd<2, f64> {
  // expected-error @+1 {{input SIMD dtype 'f32' disagrees with result SIMD dtype 'f64'}}
  %0 = meta.simd.rebind %arg0 : !meta.simd<2, f32> to !meta.simd<2, f64>
  kgen.return %0 : !meta.simd<2, f64>
}

// -----

kgen.func @meta_buffer_size(%arg0: i32) -> index {
  // expected-error @+1 {{'meta.buffer.size' op operand #0 must be parameterized buffer type, but got 'i32'}}
  %0 = meta.buffer.size %arg0 : i32
  kgen.return %0 : index
}
// -----

kgen.func @bad_buffer_rebind_size(%arg0 : !meta.buffer<42, f32>) -> !meta.buffer<1, f32> {
  // expected-error @+1 {{input buffer size '42' disagrees with result buffer size '1'}}
  %0 = meta.buffer.rebind %arg0 : !meta.buffer<42, f32> to !meta.buffer<1, f32>
  kgen.return %0 : !meta.buffer<1, f32>
}

// -----

kgen.func @bad_buffer_rebind_dtype(%arg0 : !meta.buffer<42, f32>) -> !meta.buffer<42, f64> {
  // expected-error @+1 {{input buffer dtype 'f32' disagrees with result buffer dtype 'f64'}}
  %0 = meta.buffer.rebind %arg0 : !meta.buffer<42, f32> to !meta.buffer<42, f64>
  kgen.return %0 : !meta.buffer<42, f64>
}

// -----

kgen.func @meta_buffer_dtype(%arg0: i32) -> !kgen.dtype {
  // expected-error @+1 {{'meta.buffer.dtype' op operand #0 must be parameterized buffer type, but got 'i32'}}
  %0 = meta.buffer.dtype %arg0 : i32
  kgen.return %0 : !kgen.dtype
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @only_returns<() -> p2: i4>() {
  kgen.return <p2: i4 = 2>
}

kgen.func @test_only_returns() {
  // expected-error @+1 {{caller output parameter #0 has type 'index' but callee expected type 'i4'}}
  kgen.call @only_returns<()->p2>() : () -> ()
  kgen.return
}

// -----

// expected-note @+1 {{callee declared here}}
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
kgen.generator @bad<() -> size>(%arg0: si32) -> si32 implements @itf {
  kgen.return<size = 42> %arg0 : si32
}

// -----

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf<size>()

// expected-error @+1 {{generator input parameter #0 has name "barf" but interface expected name "size"}}
kgen.generator @bad<barf>() implements @itf {
  kgen.return
}

// -----

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf<() -> result>(si32)

// expected-error @+1 {{generator result parameter #0 has name "size" but interface expected name "result"}}
kgen.generator @bad<() -> size: i8>(%arg0: si32) implements @itf {
  kgen.return<size:i8 = 42>
}

// -----

// expected-error @below {{expected attribute value}}
%0 = kgen.param.constant : i32 = <[:i32]>

// -----

kgen.generator.interface @take_and_return<p1 -> r1>()

// expected-error @+1 {{invalid cyclic reference between operations defining and using parameters}}
kgen.func @self_cyclic() {
  // Uses r1 and defines r1
  kgen.call @take_and_return<p1 = r1 -> r1>() : () -> ()
  // expected-note @-1 {{this operation uses parameter "r1", which is defined by itself}}
  kgen.return
}

// -----

kgen.generator.interface @take_and_return<p1 -> r1>()

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
kgen.generator.interface @badTypes<ty1 : dtype>(%a : !meta.scalar<ty2>)

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @callee<type: dtype>(%x: !meta.scalar<type>) {
  kgen.return
}

kgen.generator @caller<type : dtype>(%arg0: !meta.scalar<type>) {
  // expected-error @+1 {{caller input #0 has type '!meta.scalar<type>' but callee expected type '!meta.scalar<f64>'}}
  kgen.call @callee<type: dtype = f64>(%arg0) : (!meta.scalar<type>) -> ()
  kgen.return
}

// -----

// Reject uses of parameters in kgen.func since the elaborator should remove them.
kgen.func @test() {  // expected-note {{within kgen.func 'test'}}
  // Declaring parameters in a kernel is ok.
  kgen.param.declare someParam = <42>

  // Using them is not.
  // expected-error@+1 {{invalid use of parameter "someParam" in kgen.func}}
  "someop" () {} : () -> !meta.simd<someParam, f32>

  kgen.return
}

// -----

// kgen.func isn't allowed to call generators that take input parameters,
// but they are allowed to call generators with no input parameters.

kgen.generator @hasInputParam<param>() {
  kgen.return
}
kgen.generator @hasResultParam<() -> param>() {
  kgen.return<param = 42>
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
kgen.generator @region_params<r3: () -> !meta.buffer<4, dt>>() {
  kgen.return
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @takeUnary
  <unaryFn: signature<<dt: dtype>(!meta.scalar<dt>) -> !meta.scalar<dt>>>() {
  kgen.return
}

kgen.func @doubleExample(%arg0: !meta.scalar<si32>) -> !meta.scalar<si32> {
  %0 = pop.add %arg0, %arg0: !meta.scalar<si32>
  kgen.return %0 : !meta.scalar<si32>
}

kgen.generator @test_region() {
  // expected-error @+1 {{caller input parameter #0 has type}}
  kgen.call @takeUnary<
     unaryFn : (!meta.scalar<si32>) -> !meta.scalar<si32> = @doubleExample>() : () -> ()
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
  <unaryFn: signature<<dt:dtype>(!meta.scalar<dt>) -> !meta.scalar<dt>>>() {
  kgen.return
}

// expected-note @+1 {{@unary declared here}}
kgen.func @unary(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  kgen.return %arg0 : !meta.scalar<f32>
}

kgen.generator @test1() {
  // expected-error @+1 {{symbol use argument #0 has type '!meta.scalar<si32>' but @unary expected type '!meta.scalar<f32>'}}
  kgen.call @takeUnary<
     unaryFn : (!meta.scalar<si32>) -> !meta.scalar<si32> = @unary>() : () -> ()
  kgen.return
}

// -----

kgen.generator @takeUnary
  <unaryFn: signature<<dt:dtype>(!meta.scalar<dt>) -> !meta.scalar<dt>>>() {
  kgen.return
}

// expected-note @+1 {{@unary2 declared here}}
kgen.generator @unary2<dt: dtype>(%arg0: !meta.scalar<si32>) -> !meta.scalar<si32> {
  kgen.return %arg0 : !meta.scalar<si32>
}

kgen.generator @test2() {
  // expected-error @+1 {{symbol use has 0 input parameters but @unary2 expects 1}}
  kgen.call @takeUnary<
     unaryFn : (!meta.scalar<si32>) -> !meta.scalar<si32> = @unary2>() : () -> ()
  kgen.return
}

// -----

kgen.generator @call_param() {
  // expected-error @+1 {{'kgen.call_param' callee parameter type must be a region type}} 
  %0 = kgen.call_param[si32: 4]() 
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

kgen.generator @takeFn<unaryFn: signature<<abc>()->()>>() {
  kgen.return
}

// expected-note @+1 {{@thing declared here}}
kgen.generator @thing<dt>() {
  kgen.return
}

kgen.generator @test2() {
  // expected-error @+1 {{symbol use input parameter #0 has name "abc" but @thing expected name "dt"}}
  kgen.call @takeFn<unaryFn : signature<<abc>()->()> = @thing>() : () -> ()
  kgen.return
}

// -----

// expected-error @+1 {{"ty" parameter not defined in signature}}
kgen.generator @test<ty: type, p : signature<<x>(!kgen.paramref<ty>)->()>>() {
  kgen.return
}

// -----

// expected-error @+1 {{signature parameter "x" redefined}}
kgen.generator @test<ty: type, p : signature<<x,x>()->()>>
() {
  kgen.return
}

