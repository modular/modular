// RUN: kgen-opt -allow-unregistered-dialect %s -verify-diagnostics -split-input-file -o /dev/null

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

// expected-error @+1 {{kgen.dtype.constant requires !kgen.dtype type}}
kgen.param.constant: i8 = <#kgen.dtype.constant<f32>>

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
  // expected-error @+1 {{invalid use of parameter with no declaration "type"}}
  %y = "someop" () {} : () -> !pop.scalar<type>
  kgen.return
}

// -----

// expected-error @below {{'get_sizeof' operator requires two operands}}
"someop"() {a = #kgen.param.expr<get_sizeof, 1>} : () -> ()

// -----

// expected-error @below {{'get_sizeof' operand 0 should be a !kgen.mlirtype}}
"someop"() {a = #kgen.param.expr<get_sizeof, 1, 2> : !kgen.dtype} : () -> ()

// -----

#target = #kgen.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128> : !kgen.target

// expected-error @below {{'get_sizeof' should return an index}}
"someop"() {a = #kgen.param.expr<get_sizeof, #kgen.concretetype.constant<i32>, #target> : !kgen.dtype} : () -> ()

// -----

// expected-error @below {{'get_alignof' operator requires two operands}}
"someop"() {a = #kgen.param.expr<get_alignof, 1>} : () -> ()

// -----

// expected-error @below {{'get_alignof' operand 0 should be a !kgen.mlirtype}}
"someop"() {a = #kgen.param.expr<get_alignof, 1, 2> : !kgen.dtype} : () -> ()

// -----

#target = #kgen.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128> : !kgen.target

// expected-error @below {{'get_alignof' should return an index}}
"someop"() {a = #kgen.param.expr<get_alignof, #kgen.concretetype.constant<i32>, #target> : !kgen.dtype} : () -> ()

// -----

// expected-error @below {{reference to parameter "n" with incorrect type 'index'}}
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
kgen.generator @g2<>() {
  kgen.return
}

// -----

// expected-error @below {{'kgen.func' op cannot have input or result parameters}}
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

// expected-note @below {{@only_returns declared here}}
kgen.generator @only_returns<p1 -> p2>() {
  kgen.return<p1>
}

kgen.func @test_only_returns() {
  // expected-error @below {{symbol use has 0 input parameters but @only_returns expects 1}}
  kgen.call @only_returns<() -> p2 = p2>() : () -> ()
  kgen.return
}

// -----

// expected-note @below {{@only_returns declared here}}
kgen.generator @only_returns<() -> p1: i4>() {
  kgen.return <:i4 2>
}

kgen.func @test_only_returns() {
  // expected-error @below {{symbol use result parameter #0 has type 'index' but @only_returns expected type 'i4'}}
  kgen.call @only_returns<() -> p2 = p1>() : () -> ()
  kgen.return
}

// -----

kgen.generator @fn<p2>() {
  kgen.return
}

kgen.generator @input_param_name() {
  // expected-error @below {{caller input parameter #0 has name "p1" but callee expected name "p2"}}
  kgen.call @fn<p1 = 42>() : () -> ()
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
%0 = kgen.param.constant: i32 = <[:i32]>

// -----

kgen.generator.interface @take_and_return<p1 -> p2>()

// expected-error @below {{cyclic reference between expressions defining and using parameters}}
kgen.generator @self_cyclic() {
  // Uses r1 and defines r1
  // expected-note @below {{parameter "r1" is defined here, which references itself}}
  kgen.call @take_and_return<p1 = r1 -> r1 = p2>() : () -> ()
  kgen.return
}

// -----

kgen.generator.interface @take_and_return<p1 -> r1>()

// expected-error @below {{cyclic reference between expressions defining and using parameters}}
kgen.generator @mutually_recursive() {
  // Uses r2 and defines r1
  // expected-note @below {{parameter "r1" is defined here, which references the first expression}}
  kgen.call @take_and_return<p1 = r2 -> r1 = r1>() : () -> ()

  // Uses r1 and defines r2
  // expected-note @below {{parameter "r2" is defined here, which references the expression:}}
  kgen.call @take_and_return<p1 = r1 -> r2 = r1>() : () -> ()

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

// expected-error @below {{'kgen.generator.interface' op invalid use of parameter with no declaration "ty2"}}
kgen.generator.interface @badTypes<ty1 : dtype>(%a : !pop.scalar<ty2>)

// -----

// expected-note @below {{@callee declared here}}
kgen.generator @callee<type: dtype>(%x: !pop.scalar<type>) {
  kgen.return
}

kgen.generator @caller<type : dtype>(%arg0: !pop.scalar<type>) {
  // expected-error @below {{symbol use argument #0 has type '!pop.scalar<type>' but @callee expected type '!pop.scalar<f64>'}}
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
kgen.generator @nothing<() -> param>() {
  kgen.return<42>
}

kgen.func @test() {  // expected-note {{within 'kgen.func' @test}}
  // ok
  kgen.call @hasResultParam<() -> result = param>() : () -> ()

  // expected-error@+1 {{cannot reference generator with input parameters from within a concrete 'kgen.func'}}
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

// expected-error @below {{invalid use of parameter with no declaration "dt"}}
kgen.generator @region_params<r3: () -> !pop.scalar<dt>>() {
  kgen.return
}

// -----

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
  // expected-error @+1 {{@missing does not reference a KGEN declaration}}
  kgen.call @takeFn<fn: ()->() = @missing>() : () -> ()
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
  kgen.call @takeUnary<
     unaryFn : (!pop.scalar<si32>) -> !pop.scalar<si32> = @unary>() : () -> ()
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
  %0 = kgen.call_param[()->(): bind_signature(:<ty: type>()->() fn, f32)]()
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

kgen.generator @takeFn<unaryFn: <abc>()->()>() {
  kgen.return
}

// expected-note @below {{@thing declared here}}
kgen.generator @thing<dt>() {
  kgen.return
}

kgen.generator @test2() {
  // expected-error @below {{symbol use input parameter #0 has name "abc" but @thing expected name "dt"}}
  kgen.call @takeFn<unaryFn : <abc>()->() = @thing>() : () -> ()
  kgen.return
}

// -----

// expected-error @below {{nested parameter "x" redefined}}
kgen.generator @test<ty: type, p : <x, x>() -> ()>
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

// expected-note @below {{@ParamNamedA declared here}}
lit.struct.decl @ParamNamedA<A> {}

kgen.generator @give_it_B<C>() {
  // expected-error @below {{!kgen.declref symbol use input parameter #0 has name "B" but @ParamNamedA expected name "A"}}
  %0 = "a"() : () -> !kgen.declref<@ParamNamedA<B = C>>
  kgen.return
}


// -----

kgen.func @addressof_invalid_callee() {
  // expected-error @below {{@does_not_exist does not reference a KGEN declaration}}
  %0 = kgen.addressof @does_not_exist : () -> ()
  kgen.return
}

// -----

// expected-note @below {{@nullary declared here}}
kgen.func @nullary() {
  kgen.return
}

kgen.func @addressof_mismatched_signature() {
  // expected-error @below {{symbol use has 1 argument but @nullary expects 0}}
  %0 = kgen.addressof @nullary : (index) -> ()
  kgen.return
}

// -----

kgen.generator.interface @generator<size>()

// expected-note @below {{within 'kgen.func' @addressof_parametric_in_func}}
kgen.func @addressof_parametric_in_func() {
  // expected-error @below {{'kgen.addressof' op cannot reference generator with input parameters from within a concrete 'kgen.func'}}
  %0 = kgen.addressof @generator<size = 1> : () -> ()
  kgen.return
}

// -----

// expected-error @below {{@evaluator does not reference a KGEN declaration}}
kgen.generator.interface @evaluateMe(index) -> index
  evaluator (!pop.pointer<(index) -> index>, index) -> index = @evaluator<N=4>

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
    // expected-error @below {{'kgen.param.constant' op reference to parameter "SomeParam" with incorrect type 'index'}}
    %0 = kgen.param.constant = <SomeParam>
    // expected-note @below {{parameter defined with type '!kgen.dtype'}}
    kgen.param.declare SomeParam : dtype = <f32>
    kgen.return
  }
  kgen.return
}

// -----

kgen.generator @simpleEvaluator<N, FN:type>(%funcs: !pop.pointer<FN>, %size: index) -> index {
  %0 = kgen.param.constant = <N>
  kgen.return %0 : index
}

// expected-error @below {{@doesNotExist does not reference a KGEN declaration}}
kgen.generator.interface @pickFirst()
  evaluator (!pop.pointer<() -> ()>, index) -> index = @simpleEvaluator<N=0, FN:type=()->()>
  defaultImpl () -> () = @doesNotExist

// -----

kgen.generator @simpleEvaluator<N, FN:type>(%funcs: !pop.pointer<FN>, %size: index) -> index {
  %0 = kgen.param.constant = <N>
  kgen.return %0 : index
}

kgen.func @defaultFunc() {
  kgen.return
}

// expected-error @below {{defaultImpl @defaultFunc must be a generator}}
kgen.generator.interface @pickFirst()
  evaluator (!pop.pointer<() -> ()>, index) -> index = @simpleEvaluator<N=0, FN:type=()->()>
  defaultImpl () -> () = @defaultFunc

// -----

// expected-error @below {{could not find referenced symbol '@doesNotExist'}}
kgen.export @doesNotExist

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

kgen.generator @apply_error<fn: <A>() -> ()>() {
  // expected-error @below {{custom op 'kgen.param.declare' 'apply' function cannot be parametric}}
  kgen.param.declare fn = <apply(:<A>() -> () fn)>
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

// expected-error @below {{list attribute type requires 2 elements but value has 1}}
"some.op"() {a = #kgen<list[1]> : !kgen.list<index[2]>} : () -> ()

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @callee(%byval: !pop.pointer<i32> byref) {
  kgen.return
}

kgen.generator @caller(%arg: !pop.pointer<i32>) {
  // Ok
  kgen.call @callee(%arg) : (!pop.pointer<i32> byref) -> ()

  // expected-error @+1 {{symbol use metadata is #kgen.metadata<[byval], none> but @callee expected #kgen.metadata<[byref], none>}}
  kgen.call @callee(%arg) : (!pop.pointer<i32>) -> ()
  kgen.return
}

// -----

kgen.generator @target_params2<t0: target>()
 // expected-error @below {{expected '='}}
  constraints <[eq(:target t0, #kgen.target<triple"triple", "cpu", "features", 3, 4>), "must support target!!"]> {
  kgen.return
}

// -----

// expected-error @below {{'kgen.func' op can only have default value input conventions}}
kgen.func @conventions(%arg0: !pop.pointer<index> byref) {
  kgen.return
}

// -----

// expected-error @below {{custom op 'kgen.generator' a function that throws should have 1 result}}
kgen.generator @throws() throws {
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

kgen.generator @evaluator() {
  kgen.return
}

kgen.generator @noOptions() {
  // expected-error @below {{expected attribute value}}
  // expected-error @below {{expected a symbol attribute}}
  kgen.param.declare chosenImpl : () -> () = <evaluate(:() -> () :() -> ()@evaluator)>
  kgen.return
}

// -----

kgen.generator @evaluator<N>() {
  kgen.return
}

kgen.generator @f1() {
  kgen.return
}

kgen.generator @parametricEvaluator() {
  // expected-error @below {{'evaluate' evaluator cannot be parametric}}
  kgen.param.declare chosenImpl : () -> () = <evaluate(:() -> () @f1, :<N>() -> ()@evaluator)>
  kgen.return
}

// -----

kgen.generator @evaluator() -> (index, index) {
  %0 = kgen.param.constant = <0>
  %1 = kgen.param.constant = <1>
  kgen.return %0, %1 : index, index
}

kgen.generator @f1() {
  kgen.return
}

kgen.generator @multiReturnEvaluator() {
  // expected-error @below {{'evaluate' evaluator must return one result}}
  kgen.param.declare chosenImpl : () -> () = <evaluate(:() -> () @f1, :() -> (index, index) @evaluator)>
  kgen.return
}

// -----

kgen.generator @evaluator() -> index {
  kgen.return
}

kgen.generator @differentType() {
  kgen.param.declare bad = <3>
  // expected-error @below {{expected a signature type for 'evaluate'}}
  kgen.param.declare chosenImpl : () -> () = <evaluate(:index bad, :() -> index @evaluator)>
  kgen.return
}

// -----

kgen.generator @fwd<in -> out>() {
  kgen.return<in>
}

// expected-error @below {{cyclic reference between expressions}}
kgen.generator @cyclicIf() {
  kgen.param.declare cond_var: i1 = <1>
  // This outputs a single parameter that is either the result of the call
  // or N itself.
  // expected-note @below {{parameter "M2" is defined here}}
  kgen.param.if <cond_var -> M2> {
    kgen.call @fwd<in = N -> outM = out>() : () -> ()
    kgen.param.yield<outM>
  } else {
    kgen.param.yield<N>
  }
  // This forwards the output parameter of the if statement back around to N,
  // creating a cycle.
  // expected-note @below {{parameter "N" is defined here}}
  kgen.call @fwd<in = M2 -> N = out>() : () -> ()
  kgen.return
}

// -----

kgen.generator @noResultParam() {
  kgen.param.declare cond_var: i1 = <1>
  // expected-error @below {{expected a kgen.param.yield in order to return result parameters}}
  kgen.param.if <cond_var -> out> {
    kgen.param.yield<3>
  } else {
    // expected-note @below {{unknown terminator defined here}}
    hlcf.return
  }
  kgen.return
}

// -----

kgen.generator @badResultParam() {
  kgen.param.declare cond_var: i1 = <1>
  // expected-note @below {{result parameter defined here}}
  kgen.param.if <cond_var -> out> {
    // expected-error @below {{result parameter type did not match, expected 'index' but got 'i1'}}
    kgen.param.yield<:i1 1>
  } else {
    kgen.param.yield<:i1 0>
  }
  kgen.return
}

// -----

kgen.generator @declareWrongType() {
  // expected-error @below {{'kgen.param.declare' op declares a parameter with type 'index' but parameter expression has type 'i32'}}
  "kgen.param.declare"() {paramDecl = #kgen<param.decl p1 : index>, value = 1 : i32} : () -> ()
  kgen.return
}
