// RUN: kgen-opt -allow-unregistered-dialect %s -verify-parameters -verify-diagnostics -split-input-file

kgen.generator @test() {
  // expected-error @+1 {{invalid use of parameter with no declaration "p"}}
  "someop" () {
    attr = #kgen.param.decl.ref<"p"> : i1
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

// expected-error @+1 {{operator requires an index or integer type}}
%0 = kgen.param.constant: f32 = <shl(1., 2.)>

// -----

// expected-error @+1 {{integer literal not valid for specified type}}
kgen.param.constant: !kgen.dtype = <mul(1, 4)>

// -----

kgen.generator @foo() {
  // expected-error @+1 {{invalid use of parameter with no declaration "f32"}}
  kgen.param.constant: i8 = <f32>
  kgen.return
}

// -----

// expected-error @+2 {{attribute type different than expected: expected '!kgen.dtype', but got 'index'}}
kgen.generator @scalar_params_verbose<n>(%x :
           !pop.scalar<#kgen.param.decl.ref<"n"> : index>) {
  kgen.return
}

// -----

// expected-error @+1 {{invalid use of parameter with no declaration "abc"}}
kgen.generator @scalar_params_verbose(%x : !pop.scalar<abc>) {
  kgen.return
}

// -----

kgen.generator @dtype_params() {
  // expected-error @+1 {{invalid use of parameter with no declaration "T"}}
  %y = "someop" () {} : () -> !pop.scalar<T>
  kgen.return
}

// -----

// expected-error @below {{get_sizeof operator requires two operands}}
"someop"() {a = #kgen.param.expr<get_sizeof, 1>} : () -> ()

// -----

// expected-error @below {{get_sizeof operand 0 should be a type expression}}
"someop"() {a = #kgen.param.expr<get_sizeof, 1, 2> : !kgen.dtype} : () -> ()

// -----

#target = #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target

// expected-error @below {{get_sizeof should return an index or !kgen.int_literal}}
"someop"() {a = #kgen.param.expr<get_sizeof, #kgen.type<i32> : !kgen.type, #target> : !kgen.dtype} : () -> ()

// -----

// expected-error @below {{get_alignof operator requires two operands}}
"someop"() {a = #kgen.param.expr<get_alignof, 1>} : () -> ()

// -----

// expected-error @below {{get_alignof operand 0 should be a type expression}}
"someop"() {a = #kgen.param.expr<get_alignof, 1, 2> : !kgen.dtype} : () -> ()

// -----

#target = #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target

// expected-error @below {{get_alignof should return an index or !kgen.int_literal}}
"someop"() {a = #kgen.param.expr<get_alignof, #kgen.type<i32> : !kgen.type, #target> : !kgen.dtype} : () -> ()

// -----

// expected-error @below {{type of index reference #kgen.param.index.ref<0, false, 0> : index does not match parameter type 'ui32'}}
// expected-error @below {{'kgen.generator' op reference to parameter "n" with incorrect type 'index'}}
// expected-note @below {{parameter defined with type 'ui32'}}
kgen.generator @scalar_params_verbose<n : ui32>(%x : !pop.array<n, scalar<invalid>>) {
  kgen.return
}

// -----

kgen.func @entry() {
  // expected-error @below {{@undefined does not reference a KGEN declaration}}
  kgen.call @undefined() : () -> ()
  kgen.return
}

// -----

module @nested_nondecl {
}

kgen.func @entry() {
  // expected-error @below {{@nested_nondecl::@undefined does not reference a KGEN declaration}}
  kgen.call @nested_nondecl::@undefined() : () -> ()
  kgen.return
}

// -----

kgen.generator @g1(%x : i32) {
  // expected-error @below {{symbol use has 1 argument but @g2 expects 0}}
  kgen.call @g2(%x) : (i32) -> ()
  kgen.return
}

// expected-note @below {{@g2 declared here}}
kgen.generator @g2() {
  kgen.return
}

// -----

// expected-note @below {{@only_returns declared here}}
kgen.generator @only_returns<p1>() {
  kgen.return
}

kgen.func @test_only_returns() {
  // expected-error @below {{symbol use has 0 input parameters but @only_returns expects 1}}
  kgen.call @only_returns() : () -> ()
  kgen.return
}

// -----

// expected-note @below {{@fn declared here}}
kgen.generator @fn(%a: i1) -> i1 {
  kgen.return %a : i1
}

kgen.func @result_type(%a: i1) {
  // expected-error @below {{symbol use result #0 has type 'f32' but @fn expected type 'i1'}}
  kgen.call @fn(%a) : (i1) -> f32
  kgen.return
}

// -----

// expected-error @below {{expected attribute value}}
%0 = kgen.param.constant: i32 = <[:i32]>

// -----

// expected-error @below {{cyclic reference between expressions defining and using parameters}}
kgen.generator @self_cyclic() {
  // Uses r1 and defines r1
  // expected-note @below {{parameter "r1" is defined here, which references itself}}
  kgen.param.declare r1 = <r1>
  kgen.return
}

// -----


// expected-error @below {{cyclic reference between expressions defining and using parameters}}
kgen.generator @mutually_recursive() {
  // Uses r2 and defines r1
  // expected-note @below {{parameter "r1" is defined here, which references the first expression}}
  kgen.param.declare r1 = <r2>

  // Uses r1 and defines r2
  // expected-note @below {{parameter "r2" is defined here, which references the expression:}}
  kgen.param.declare r2 = <r1>

  kgen.return
}

// -----

// expected-error @below {{cyclic reference between expressions defining and using parameters}}
kgen.generator @use_itself() {
  // expected-note @below {{parameter "F" is defined here, which references itself}}
  kgen.param.declare.region F = (){
    kgen.call_param[() -> (): F]()
    kgen.return
  }
  kgen.return
}

// -----

// expected-error @below {{cyclic reference between expressions defining and using parameters}}
kgen.generator @region_cycle() {
  // expected-note @below {{parameter "F" is defined here, which references the first expression}}
  kgen.param.declare.region F = () -> index {
    %0 = kgen.param.constant = <N>
    kgen.return %0 : index
  }
  // expected-note @below {{parameter "N" is defined here, which references the expression:}}
  kgen.param.declare N = <apply(:() -> index F)>
  kgen.return
}

// -----

// expected-error @below {{'kgen.generator' op invalid use of parameter with no declaration "ty2"}}
kgen.generator @badTypes<ty1 : dtype>(%a : !pop.scalar<ty2>) {
  kgen.return
}

// -----

// expected-note @below {{@callee declared here}}
kgen.generator @callee<DT: dtype>(%x: !pop.scalar<DT>) {
  kgen.return
}

kgen.generator @caller<DT: dtype>(%arg0: !pop.scalar<DT>) {
  // expected-error @below {{symbol use argument #0 has type '!pop.scalar<DT>' but @callee expected type '!pop.scalar<f64>'}}
  kgen.call @callee<:dtype f64>(%arg0) : (!pop.scalar<DT>) -> ()
  kgen.return
}

// -----

// kgen.func isn't allowed to call generators that take parameters,
// but they are allowed to call generators with no parameters.

kgen.generator @hasInputParam<param>() {
  kgen.return
}

kgen.func @test() {  // expected-note {{within 'kgen.func' @test}}
  // expected-error@+1 {{cannot reference generator with input parameters from within a concrete 'kgen.func'}}
  kgen.call @hasInputParam<42>() : () -> ()

  kgen.return
}

// -----

// expected-error @below {{invalid use of parameter with no declaration "dt"}}
kgen.generator @region_params<r3: () -> !pop.scalar<dt>>() {
  kgen.return
}

// -----

kgen.generator @takeUnary
  <unaryFn: <dtype>(!pop.scalar<*(0,0)>) -> !pop.scalar<*(0,0)>>() {
  kgen.return
}

kgen.func @doubleExample(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  %0 = pop.add %arg0, %arg0: !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

kgen.generator @test_region() {
  // expected-error @+1 {{caller input parameter #0 has type}}
  kgen.call @takeUnary<:(!pop.scalar<si32>) -> !pop.scalar<si32> @doubleExample>() : () -> ()
  kgen.return
}

// -----

kgen.generator @takeFn<fn: () -> ()>() {
  kgen.return
}
kgen.generator @test() {
  // expected-error @+1 {{@missing does not reference a KGEN declaration}}
  kgen.call @takeFn<:()->() @missing>() : () -> ()
  kgen.return
}

// -----

kgen.generator @takeUnary
  <unaryFn: (!pop.scalar<si32>) -> !pop.scalar<si32>>() {
  kgen.return
}

// expected-note @below {{@unary declared here}}
kgen.func @unary(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  kgen.return %arg0 : !pop.scalar<f32>
}

kgen.generator @test1() {
  // expected-error @below {{symbol use argument #0 has type '!pop.scalar<si32>' but @unary expected type '!pop.scalar<f32>'}}
  kgen.call @takeUnary<:(!pop.scalar<si32>) -> !pop.scalar<si32> @unary>() : () -> ()
  kgen.return
}

// -----

kgen.generator @takeUnary
  <unaryFn: (!pop.scalar<si32>) -> !pop.scalar<si32>>() {
  kgen.return
}

// expected-note @below {{@unary2 declared here}}
kgen.generator @unary2<dt: dtype>(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  kgen.return %arg0 : !pop.scalar<si32>
}

kgen.generator @test2() {
  // expected-error @below {{symbol use has 0 input parameters but @unary2 expects 1}}
  kgen.call @takeUnary<:(!pop.scalar<si32>) -> !pop.scalar<si32> @unary2>() : () -> ()
  kgen.return
}

// -----

kgen.generator @call_param() {
  // expected-error @+1 {{'kgen.call_param' callee parameter type must be a signature type}}
  %0 = kgen.call_param[si32: 4]()
  kgen.return
}


// -----

kgen.generator @call_param<fn: <type>()->()>() {
  // expected-error @+1 {{cannot name an operation with no results}}
  %0 = kgen.call_param[()->(): bind_signature(:<type>()->() fn, f32)]()
  kgen.return
}

// -----

kgen.func @trivial(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

kgen.func @call_param_in_func(%arg0: si32) -> si32 {
  // expected-error @below {{'kgen.call_param' op is only allowed in generators pre-elaboration}}
  %0 = kgen.call_param[(si32) -> si32: @trivial](%arg0)
  kgen.return %0: si32
}

// -----

kgen.generator @bar<F>() {
  kgen.param.declare.region fn = () {
    // expected-error @below {{'kgen.param.constant' op invalid use of parameter with no declaration "Q"}}
    %0 = kgen.param.constant = <Q>
    kgen.return
  }
  kgen.return
}

// -----

kgen.generator @doIt<SomeParam>() {
  kgen.param.declare.region fn = () {
    // expected-error @below {{'kgen.param.constant' op reference to parameter "SomeOtherParam" with incorrect type 'index'}}
    %0 = kgen.param.constant = <SomeOtherParam>
    // expected-note @below {{parameter defined with type '!kgen.dtype'}}
    kgen.param.declare SomeOtherParam : dtype = <f32>
    kgen.return
  }
  kgen.return
}

// -----

kgen.generator @apply_error() {
  // expected-error @below {{custom op 'kgen.param.declare' expected a signature type for 'apply'}}
  kgen.param.declare fn = <apply(5, 5)>
}

// -----

kgen.generator @apply_error() {
  // expected-error @below {{custom op 'kgen.param.declare' 'apply' expected a function parameter}}
  kgen.param.declare fn = <apply()>
}

// -----

kgen.generator @apply_error<fn: <index>() -> ()>() {
  // expected-error @below {{custom op 'kgen.param.declare' 'apply' function cannot be parametric}}
  kgen.param.declare fn = <apply(:<index>() -> () fn)>
}

// -----

kgen.generator @apply_error<fn: () -> ()>() {
  // expected-error @below {{custom op 'kgen.param.declare' 'apply' function must return one result}}
  kgen.param.declare fn = <apply(:() -> () fn)>
}

// -----

kgen.generator @apply_error<fn: () -> ()>() {
  // expected-error @below {{custom op 'kgen.param.declare' 'apply' function result type must be 'index' but got '!kgen.dtype'}}
  kgen.param.declare fn = <apply(:() -> (!kgen.dtype) fn)>
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @callee(%owned: !kgen.pointer<i32> inout) {
  kgen.return
}

kgen.generator @caller(%arg: !kgen.pointer<i32> owned) {
  // Ok
  kgen.call @callee(%arg) : (!kgen.pointer<i32> inout) -> ()

  // expected-error @+1 {{symbol use argument #0 has convention owned but @callee expected convention inout}}
  kgen.call @callee(%arg) : (!kgen.pointer<i32> owned) -> ()
  kgen.return
}

// -----

kgen.generator @target_params2<t0: target>() {
 // expected-error @below {{expected '='}}
  kgen.param.assert <eq(:target t0, #kgen.target<triple"triple", "cpu", "features", 3, 4>)>, "must support target!!"
  kgen.return
}

// -----

// COM: Make sure these don't crash and emit an error gracefully.

kgen.generator @no_return() {
  // expected-error @below {{block with no terminator}}
  kgen.param.declare A = <1>
}

// -----

kgen.func @no_return() {
  // expected-error @below {{block with no terminator}}
  kgen.param.declare A = <1>
}

// -----

// expected-error @below {{cyclic reference between expressions}}
kgen.generator @cyclicIf() {
  kgen.param.declare cond_var: i1 = <1>
  // expected-note @below {{parameter "M2" is defined here}}
  kgen.param.declare.region M2 = () -> index {
    kgen.param.constant = <N>
    kgen.unreachable
  }
  // This forwards the output parameter of the if statement back around to N,
  // creating a cycle.
  // expected-note @below {{parameter "N" is defined here}}
  kgen.param.declare N = <apply(:() -> index M2)>
  kgen.return
}

// -----

kgen.generator @declareWrongType() {
  // expected-error @below {{'kgen.param.declare' op declares a parameter with type 'index' but parameter expression has type 'i32'}}
  "kgen.param.declare"() {paramDecl = #kgen<param.decl p1 : index>, value = 1 : i32} : () -> ()
  kgen.return
}

// -----

kgen.generator @duplicate_decl<() -> out>() {
  // expected-note @below {{previous declaration here}}
  kgen.param.declare a = <5>
  // expected-error @below {{redeclaration of parameter "a"}}
  kgen.param.declare a = <6>
  kgen.return
}

// -----

// expected-note @below {{previous declaration here}}
kgen.generator @name_shadwing_1<a>() {
  // expected-error @below {{redeclaration of parameter "a"}}
  kgen.param.declare.region fn = <a>() {
    kgen.unreachable
  }
  kgen.return
}

// -----

kgen.generator @name_shadwing_2<a>() {
  // expected-note @below {{previous declaration here}}
  kgen.param.declare b = <a>
  kgen.param.declare.region fn = () {
    // expected-error @below {{redeclaration of parameter "b"}}
    kgen.param.declare b = <a>
    kgen.unreachable
  }
  kgen.return
}

// -----

// expected-error @below {{parameter "out" has no definition}}
kgen.generator @missing_def<() -> out>() {
  kgen.return
}

// -----

kgen.generator @bad_index_ref() {
  // expected-error @below {{index reference has no contextual signature}}
  kgen.param.declare a = <*(0,0)>
  kgen.return
}

// -----

// expected-error @below {{index reference depth 1 exceeds depth of contextual signatures: 1}}
kgen.generator @bad_index_ref<fn: <index>(!pop.array<*(1,0), i32>) -> ()>() {
  kgen.return
}

// -----

// expected-error-re @below {{index reference 1 is out of bounds: referenced signature {{.*}} has 1 input parameters}}
kgen.generator @bad_index_ref<fn: <index>(!pop.array<*(0,1), i32>) -> ()>() {
  kgen.return
}

// -----

// expected-error @below {{type of index reference #kgen.param.index.ref<0, false, 0> : index does not match parameter type 'i32'}}
kgen.generator @bad_index_ref<fn: <i32, !pop.array<*(0,0), i32>>() -> ()>() {
  kgen.return
}

// -----

kgen.func @stage_closure() {
  // expected-error @below {{staged closures cannot have parameters}}
  %0 = kgen.stage_closure = <n : ui32>() capturing -> index {
  } { name = "k" }
}

// -----

kgen.generator @variadic_get() {
  // expected-error @below {{custom op 'kgen.param.constant' 'variadic_get' expected first operand to be a variadic value}}
  kgen.param.constant = <variadic_get(:si32 2, 1)>
}

// -----

kgen.generator @variadic_get() {
  // expected-error @below {{'variadic_get' expected two operands}}
  kgen.param.constant = <#kgen.param.expr<variadic_get>>
}

// -----

kgen.generator @variadic_get() {
  // expected-error @below {{'variadic_get' expected second operand to be an index}}
  kgen.param.constant = <#kgen.param.expr<variadic_get, #kgen.variadic<> : !kgen.variadic<si32>, "foo">>
}

// -----

kgen.generator @variadic_get() {
  // expected-error @below {{custom op 'kgen.param.constant' 'variadic_get' result type should be variadic element type: expected 'si32' but got 'index'}}
  kgen.param.constant = <variadic_get(:variadic<si32> [], 1)>
}

// -----

kgen.generator @bad_return() -> index {
  // expected-error @below {{'kgen.return' op expected 1 operands, but given 0}}
  kgen.return
}

// -----

// expected-error @below {{'kgen.global' op constructor @global_ctor does not reference a function with zero arguments and zero results}}
kgen.global @global_var : i32 [@global_ctor, @global_dtor](2)

// -----

kgen.func @global_ctor() {
  kgen.return
}

kgen.func @global_dtor(%arg0: i32) -> i32 {
  kgen.return %arg0 : i32
}

// expected-error @below {{'kgen.global' op destructor @global_dtor does not reference a function with zero arguments and zero results}}
kgen.global @global_var : i32 [@global_ctor, @global_dtor](2)

// -----

// expected-error @below {{environment value "value" is an integer not of `index` type}}
module attributes {kgen.env = #kgen.env<{value = 1}>} {}


// -----

// expected-error @below {{environment value "str" is a string not of `!kgen.string` type}}
module attributes {kgen.env = #kgen.env<{str = "hello"}>} {}

// -----

// expected-error @below {{environment value "fp" is neither an index, string, or unit attribute}}
module attributes {kgen.env = #kgen.env<{fp = 2.0}>} {}

// -----

kgen.generator @variant_constant<value: i32>() {
  // expected-error @below {{variant attribute value type 'i32' does not match type at index 0 which is 'f32'}}
  %0 = kgen.param.constant: variant<f32, f64> = <#kgen.variant<:i32 value, 0>>
}

// -----

// expected-error @below {{cannot create pack with parametric element types}}
"kgen.pack.create"() : () -> !kgen.pack<T>

// -----

// expected-error @below {{expected 1 operands, but got 0}}
"kgen.pack.create"() : () -> !kgen.pack<[index]>

// -----

kgen.func @pack(%arg0: i32) {
  // expected-error @below {{operand #0 should have type 'index' but got 'i32'}}
  "kgen.pack.create"(%arg0) : (i32) -> !kgen.pack<[index]>
  kgen.return
}

// -----

// expected-error @below {{'byref_result' argument must be the last argument}}
kgen.func @invalid(%arg0: !kgen.pointer<index> byref_result, %arg1: index) -> !kgen.none {
  kgen.unreachable
}

// -----

kgen.generator @two_params<a, b>() {
  // expected-error @below {{callee expects 2 parameters but only got 1}}
  kgen.param.declare f: <index, index>() -> () = <@two_params<?>>
  kgen.return
}

// -----

kgen.generator @kernel() {
  kgen.return
}

kgen.generator export @top() {
  // expected-error @below {{custom op 'kgen.param.constant' the emission kind must be either llvm or asm}}
  kgen.param.constant: string = <compile_assembly(current_target(), something, 0, :() -> () @kernel)>
  kgen.return
}

// -----

// expected-error @below {{!kgen.source_struct parameter type mismatch at index 0. Expected '!kgen.dtype', got 'index'}}
kgen.func @illegal_source_struct_param_type(%arg0: !kgen.source_struct<"Foo"[dt: dtype]<:index 8>>) {}

// -----

// expected-error @below {{!kgen.source_struct parameter decl and parameter value length mismatch. Expected 1, got 0}}
kgen.func @illegal_source_struct_param_length(%arg0: !kgen.source_struct<"Foo"[dt: dtype]>) {}
