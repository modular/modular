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

// expected-error @below {{'get_sizeof' should return an index}}
"someop"() {a = #kgen.param.expr<get_sizeof, #kgen.concretetype.constant<i32>, #kgen.target<host>> : !kgen.dtype} : () -> ()

// -----

// expected-error @below {{'get_alignof' operator requires two operands}}
"someop"() {a = #kgen.param.expr<get_alignof, 1>} : () -> ()

// -----

// expected-error @below {{'get_alignof' operand 0 should be a !kgen.mlirtype}}
"someop"() {a = #kgen.param.expr<get_alignof, 1, 2> : !kgen.dtype} : () -> ()

// -----

// expected-error @below {{'get_alignof' should return an index}}
"someop"() {a = #kgen.param.expr<get_alignof, #kgen.concretetype.constant<i32>, #kgen.target<host>> : !kgen.dtype} : () -> ()

// -----

// expected-note @+2 {{parameter defined with type 'ui32'}}
// expected-error @+1 {{reference to parameter "n" with incorrect type 'index'}}
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

// expected-note @below {{@only_returns declared here}}
kgen.generator @only_returns<p1 -> index>() {
  kgen.return<p1>
}

kgen.func @test_only_returns() {
  // expected-error @below {{symbol use has 0 input parameters but @only_returns expects 1}}
  kgen.call @only_returns<()->p2>() : () -> ()
  kgen.return
}

// -----

// expected-note @below {{@only_returns declared here}}
kgen.generator @only_returns<() -> i4>() {
  kgen.return <:i4 2>
}

kgen.func @test_only_returns() {
  // expected-error @below {{symbol use result parameter #0 has type 'index' but @only_returns expected type 'i4'}}
  kgen.call @only_returns<()->p2>() : () -> ()
  kgen.return
}

// -----

kgen.generator @fn<p2>() {
  kgen.return
}

kgen.func @input_param_name() {
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
  <unaryFn: <dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt>>() {
  kgen.return
}

// expected-note @below {{@unary declared here}}
kgen.func @unary(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  kgen.return %arg0 : !pop.scalar<f32>
}

kgen.generator @test1() {
  // expected-error @below {{symbol use argument #0 has type '!pop.scalar<si32>' but @unary expected type '!pop.scalar<f32>'}}
  // expected-error @below {{caller input parameter #0 has type '!kgen.signature<[], [], (!pop.scalar<si32>) -> !pop.scalar<si32>>' but callee expected type '!kgen.signature<[dt : !kgen.dtype], [], (!pop.scalar<dt>) -> !pop.scalar<dt>>'}}
  kgen.call @takeUnary<
     unaryFn : (!pop.scalar<si32>) -> !pop.scalar<si32> = @unary>() : () -> ()
  kgen.return
}

// -----

kgen.generator @takeUnary
  <unaryFn: <dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt>>() {
  kgen.return
}

// expected-note @below {{@unary2 declared here}}
kgen.generator @unary2<dt: dtype>(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  kgen.return %arg0 : !pop.scalar<si32>
}

kgen.generator @test2() {
  // expected-error @below {{caller input parameter #0 has type '!kgen.signature<[], [], (!pop.scalar<si32>) -> !pop.scalar<si32>>' but callee expected type '!kgen.signature<[dt : !kgen.dtype], [], (!pop.scalar<dt>) -> !pop.scalar<dt>>'}}
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
  %0 = kgen.call_param[<ty: type>()->(): fn]<ty : !kgen.mlirtype = f32>()
  kgen.return
}

// -----

kgen.generator @call_param<fn: <ty: type>()->()>() {
  // expected-error @below {{custom op 'kgen.call_param' caller input parameter #0 has type 'index' but callee expected type '!kgen.mlirtype'}}
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
  kgen.call @signature_taking_callee<fn: <size>() -> () = region>() : () -> ()
  // expected-error @below {{body region didn't have a kgen.return op?}}
  fn<size>(%arg0: i32) {}
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

// expected-error @below {{signature mismatches body}}
kgen.struct.decl @StructReturns<() -> dtype> {
}

// -----

// expected-error @below {{expected declaration body to have no arguments}}
"kgen.struct.decl"() ({
^bb0(%arg0: i32):
}) {sym_name = "StructArgs", constraints = #kgen<constraints[]>,
    signature = !kgen.signature<[], [], () -> ()>
    } : () -> ()

// -----

kgen.generator @target_params<t0: i1, t1: i1>()
  // expected-error @+1 {{custom op 'kgen.generator' target_supports only allowed on target types}}
  constraints <[target_supports(:i1 t0, t1), "must support target!!"]> {
  kgen.return
}

// -----

kgen.generator @target_params<t0: i1, t1: i1>()
  // expected-error @+1 {{custom op 'kgen.generator' target_supports must have two operands}}
  constraints <[target_supports(:i1 t0), "must support target!!"]> {
  kgen.return
}

// -----

kgen.struct.decl @StructDuplicate {
  // expected-note @below {{see previous declaration here}}
  kgen.struct.field x : i32
  kgen.struct.field y : i32
  // expected-error @below {{duplicate struct field "x"}}
  kgen.struct.field x : i32
}

// -----

kgen.struct.decl @SomeType<v, b> {}

// expected-error @below {{invalid use of parameter with no declaration "c"}}
kgen.generator.interface @InvalidTypeParamValue<a>() ->
    !kgen.declref<@SomeType<v = a, b = c>>

// -----

// expected-note @below {{@SomeType declared here}}
kgen.struct.decl @SomeType<v, d> {}

// expected-error @below {{!kgen.declref symbol use input parameter #1 has name "b" but @SomeType expected name "d"}}
kgen.generator.interface @InvalidTypeParamValue<a, c>() ->
    !kgen.declref<@SomeType<v = a, b = c>>

// -----

kgen.struct.decl @Bar<a: type> {
  kgen.struct.field x : !pop.array<32, a>
}

kgen.generator @invalid_field_type<c: type>(%a: !kgen.paramref<c>) {
  // expected-error @below {{perand #0 has type '!kgen.paramref<c>' but corresponding struct field "x" expected '!pop.array<32, a>'}}
  %0 = kgen.struct.create(%a) : (!kgen.paramref<c>) -> !kgen.declref<@Bar<a: type = index>>
  kgen.return
}

// -----

kgen.struct.decl @Bar {}

kgen.generator @invalid_field_name(%a: index, %container: !kgen.declref<@Bar>) {
  // expected-error @below {{struct @Bar has no field named "a"}}
  %0 = kgen.struct.insert %a, %container[a] : index into !kgen.declref<@Bar>
  kgen.return
}

// -----

kgen.struct.decl @Bar {
  kgen.struct.field a : i32
}

kgen.generator @invalid_field_name(%a: index, %container: !kgen.declref<@Bar>) {
  // expected-error @below {{cannot insert value of type 'index' into struct field "a" which expected 'i32'}}
  %0 = kgen.struct.insert %a, %container[a] : index into !kgen.declref<@Bar>
  kgen.return
}

// -----

kgen.struct.decl @Bar {}

kgen.generator @invalid_field_name(%a: index, %container: !kgen.declref<@Bar>) {
  // expected-error @below {{struct @Bar has no field named "a"}}
  %0 = kgen.struct.extract %container[a] : index from !kgen.declref<@Bar>
  kgen.return
}

// -----

// expected-note @below {{@ParamNamedA declared here}}
kgen.struct.decl @ParamNamedA<A> {}

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

// expected-note @below {{within kgen.func 'addressof_parametric_in_func'}}
kgen.func @addressof_parametric_in_func() {
  // expected-error @below {{cannot reference generator with input arguments from concrete kgen.func}}
  %0 = kgen.addressof @generator<size = 1> : () -> ()
  kgen.return
}

// -----

// expected-error @below {{@evaluator does not reference a KGEN declaration}}
kgen.generator.interface @evaluateMe(index) -> index
  evaluator (!pop.pointer<(index) -> index>, index) -> index = @evaluator<N=4>

// -----

// expected-note @below {{referenced evaluator declared here}}
kgen.generator.interface @evaluator<N>(
    %funcs: (index) -> index, %size: index) -> index

// expected-error @below {{interface evaluator argument #0 has type '!pop.pointer<(index) -> index>' but referenced evaluator expected type '(index) -> index'}}
kgen.generator.interface @evaluateMe(index) -> index
  evaluator ((index) -> index, index) -> index = @evaluator<N=4>

// -----

kgen.generator @foo<fn: () -> ()>() {
  kgen.return
}

kgen.generator @bar<F>() {
  kgen.call @foo<fn: () -> () = region>() : () -> ()
  fn() {
    // expected-error @below {{'kgen.param.constant' op invalid use of parameter with no declaration "Q"}}
    %0 = kgen.param.constant = <Q>
    kgen.return
  }
  kgen.return
}

// -----

kgen.generator @foo<fn: ()->() -> index>() {
  kgen.return<10>
}

// expected-error @below {{invalid cyclic reference between operations defining and using parameters}}
kgen.generator @baz<F>() {
  // expected-note @below {{this operation uses parameter "B", which is defined by the first operation}}
  kgen.call @foo<fn:()->()=region -> kValue>() : ()->()
  fn() {
    %1 = kgen.param.constant = <B>
    kgen.return
  }
  // expected-note @below {{this operation uses parameter "kValue", which is defined by:}}
  kgen.param.declare B = <add(F, kValue)>
  kgen.return
}

// -----

kgen.generator @callMe<fn: ()->()>() {
  kgen.return
}

kgen.generator @doIt<SomeParam>() {
  kgen.call @callMe<fn: ()->() = region>() : () -> ()
  fn() {
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

// expected-error @below {{exports must not be empty}}
kgen.export []

// -----

// expected-error @below {{could not find referenced symbol '@doesNotExist'}}
kgen.export [@doesNotExist]

// -----

kgen.generator @call_indirect_parametric(%arg0: !kgen.signature<[N : index], [], () -> ()>) {
  // expected-error @below {{'kgen.call_indirect' op requires the signature callee to have no input or output parameters}}
  // expected-note @below {{use `bind_signature`}}
  "kgen.call_indirect"(%arg0) : (!kgen.signature<[N : index], [], () -> ()>) -> ()
  kgen.return
}

// -----

kgen.generator @partial_apply(%arg0: !kgen.signature<[], [], () -> ()>) {
  // expected-error @below {{'kgen.partial_apply' op expected indices to be sorted ascending}}
  "kgen.partial_apply"(%arg0) {boundInputs = array<i64: 1, 0>} : (!kgen.signature<[], [], () -> ()>) -> !kgen.signature<[], [], () -> ()>
  kgen.return
}

// -----

kgen.generator @partial_apply(%arg0: !kgen.signature<[], [], () -> ()>, %arg1: i32) {
  // expected-error @below {{'kgen.partial_apply' op mismatch between number of indices and inputs: 0 vs 1}}
  "kgen.partial_apply"(%arg0, %arg1) {boundInputs = array<i64>} : (!kgen.signature<[], [], () -> ()>, i32) -> !kgen.signature<[], [], () -> ()>
  kgen.return
}

// -----

kgen.generator @partial_apply(%arg0: !kgen.signature<[], [], () -> ()>, %arg1: i32) {
  // expected-error @below {{'kgen.partial_apply' op bound input index is out of range: 0}}
  "kgen.partial_apply"(%arg0, %arg1) {boundInputs = array<i64: 0>} : (!kgen.signature<[], [], () -> ()>, i32) -> !kgen.signature<[], [], () -> ()>
  kgen.return
}

// -----

kgen.generator @partial_apply(%arg0: !kgen.signature<[], [], (i32, i32) -> ()>, %arg1: i32, %arg2: i32) {
  // expected-error @below {{'kgen.partial_apply' op duplicate bound input index: 0}}
  "kgen.partial_apply"(%arg0, %arg1, %arg2) {boundInputs = array<i64: 0, 0>} : (!kgen.signature<[], [], (i32, i32) -> ()>, i32, i32) -> !kgen.signature<[], [], (i32, i32) -> ()>
  kgen.return
}

// -----

kgen.generator @partial_apply(%arg0: !kgen.signature<[], [], (i32) -> ()>, %arg1: i64) {
  // expected-error @below {{'kgen.partial_apply' op input bound to argument #0}}
  "kgen.partial_apply"(%arg0, %arg1) {boundInputs = array<i64: 0>} : (!kgen.signature<[], [], (i32) -> ()>, i64) -> !kgen.signature<[], [], () -> ()>
  kgen.return
}

// -----

kgen.generator @partial_apply(%arg0: !kgen.signature<[], [], (i16, i32, i64) -> ()>, %arg1: i32) {
  // expected-error @below {{'kgen.partial_apply' op result signature does not match}}
  "kgen.partial_apply"(%arg0, %arg1) {boundInputs = array<i64: 1>} : (!kgen.signature<[], [], (i16, i32, i64) -> ()>, i32) -> !kgen.signature<[], [], (i32, i64) -> ()>
  kgen.return
}

// -----

kgen.generator @partial_apply_syntax(%arg0: !kgen.signature<[], [], (i8) -> ()>) {
  // expected-error @below {{custom op 'kgen.partial_apply' expected '?' or an operand in binding list}}
  kgen.partial_apply %arg0([])
  kgen.return
}

// -----

kgen.generator @partial_apply_syntax(%arg0: !kgen.signature<[], [], (i8) -> ()>, %arg1: i8, %arg2: i8) {
  // expected-error @below {{custom op 'kgen.partial_apply' there are more bound inputs than arguments}}
  kgen.partial_apply %arg0(%arg1, %arg2) : (i8) -> ()
  kgen.return
}

// -----

kgen.generator @iterate_not_a_list_type(%list: !kgen.list<index[3]>) {
  // expected-error @below {{custom op 'kgen.list.iterate' expected a list type}}
  kgen.list.iterate %v in %list : i32 [0 : (d0, len) -> (d0 + 1)]
  kgen.return
}

// -----

kgen.generator @iterate_wrong_result_type_count(%list: !kgen.list<index[3]>) {
  // expected-error @below {{custom op 'kgen.list.iterate' expected the same number of result types as arguments: 0 but got 1}}
  kgen.list.iterate %v in %list : list<index[3]> [0 : (d0, len) -> (d0 + 1)] () -> i32
}

// -----

kgen.generator @iterate_wrong_region_arg_count(%list: !kgen.list<index[3]>) {
  // expected-error @below {{'kgen.list.iterate' op expected the number of region arguments to match the number of indices plus the number of loop-carried values}}
  "kgen.list.iterate"(%list) ({
    kgen.list.yield
  }) {
    map = affine_map<(d0, len) -> (d0 + 1)>, init = #kgen<exprs[1 : index]>
  } : (!kgen.list<index[3]>) -> ()
}

// -----

kgen.generator @iterate_wrong_region_arg_count(%list: !kgen.list<index[3]>) {
  // expected-error @below {{'kgen.list.iterate' op expected first 1 argument types to be list element type 'index'}}
  "kgen.list.iterate"(%list) ({
  ^bb0(%arg0: i32):
    kgen.list.yield
  }) {
    map = affine_map<(d0, len) -> (d0 + 1)>, init = #kgen<exprs[1 : index]>
  } : (!kgen.list<index[3]>) -> ()
}

// -----

kgen.generator @iterate_wrong_region_arg_count(%list: !kgen.list<index[3]>, %arg: i32) {
  // expected-error @below {{'kgen.list.iterate' op expected last 1 argument types to be equal to the initial value types}}
  %0 = "kgen.list.iterate"(%list, %arg) ({
  ^bb0(%arg0: index, %arg1: i64):
    kgen.list.yield
  }) {
    map = affine_map<(d0, len) -> (d0 + 1)>, init = #kgen<exprs[1 : index]>
  } : (!kgen.list<index[3]>, i32) -> i32
}

// -----

kgen.generator @iterate_wrong_result_type_count(%list: !kgen.list<index[3]>) {
  // expected-error @below {{'kgen.list.iterate' op expected map to have 1 variable inputs}}
  kgen.list.iterate %v in %list : list<index[3]> [0 : () -> (1)] {
    kgen.list.yield
  }
  kgen.return
}

// -----

kgen.generator @iterate_wrong_result_type_count(%list: !kgen.list<index[3]>) {
  // expected-error @below {{'kgen.list.iterate' op expected map to have 1 results}}
  kgen.list.iterate %v in %list : list<index[3]> [0 : (d0) -> (d0 + 1, d0)] {
    kgen.list.yield
  }
  kgen.return
}

// -----

kgen.generator @iterate_wrong_result_type_count(%list: !kgen.list<index[3]>) {
  // expected-error @below {{'kgen.list.iterate' op expected map to have 0 symbolic inputs}}
  kgen.list.iterate %v in %list : list<index[3]> [0 : (d0)[s0] -> (d0 + 1)] {
    kgen.list.yield
  }
  kgen.return
}

// -----

// expected-error @below {{too many input declarations}}
"someop"() {f = #kgen.expr.func<(A, B) -> A> : !kgen.signature<[], [], (index) -> index>} : () -> ()

// -----

// expected-error @below {{not enough input declarations}}
"someop"() {f = #kgen.expr.func<() -> A> : !kgen.signature<[], [], (index) -> index>} : () -> ()

// -----

// expected-error @below {{"B" parameter not defined in function}}
"someop"() {f = #kgen.expr.func<(A) -> B> : !kgen.signature<[], [], (index) -> index>} : () -> ()

// -----

// expected-error @below {{use of "B" with incorrect type in function}}
"someop"() {f = #kgen.expr.func<(A) -> add(:i32 B, B)> : !kgen.signature<[B : index], [], (index) -> i32>} : () -> ()

// -----

// expected-error @below {{too many result expressions}}
"someop"() {f = #kgen.expr.func<(A) -> (A, B)> : !kgen.signature<[B : index], [], (index) -> i32>} : () -> ()

// -----

// expected-error @below {{not enough result expressions}}
"someop"() {f = #kgen.expr.func<(A) -> ()> : !kgen.signature<[B : index], [], (index) -> i32>} : () -> ()

// -----

kgen.func @list_index_out_of_bounds(%list : !kgen.list<index[0]>) {
  // expected-error @below {{'kgen.list.get' op list index out-of-range}}
  %0 = kgen.list.get %list[0] : <index[0]>
  kgen.return
}

// -----

kgen.generator @apply_error() {
  // expected-error @below {{custom op 'kgen.param.declare' expected a signature type for operator}}
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

kgen.generator @apply_error<fn: () -> index>() {
  // expected-error @below {{custom op 'kgen.param.declare' 'apply' function expected 0 inputs but got 1}}
  kgen.param.declare fn = <apply(:() -> index fn, 1)>
}

// -----

kgen.generator @apply_error<fn: (index) -> index>() {
  // expected-error @below {{custom op 'kgen.param.declare' 'apply' input #0 is '!kgen.dtype' but function expected 'index'}}
  kgen.param.declare fn = <apply(:(index) -> index fn, :dtype f32)>
}

// -----

kgen.generator @iterate<next: (index) -> index, cond: (index) -> i1>() {
  // expected-error @below {{custom op 'kgen.iterate' not enough init values}}
  kgen.iterate (I) in [(), next, cond] {
    kgen.return
  }
}

// -----

kgen.generator @iterate<next: (index) -> index, cond: (index) -> i1>() {
  // expected-error @below {{custom op 'kgen.iterate' too many init values}}
  kgen.iterate (I) in [(0, 1), next, cond] {
    kgen.return
  }
}

// -----

kgen.generator @iterate<next: (index) -> index, cond: (index) -> i1>() {
  // expected-error @below {{'kgen.iterate' op body results should match argument types}}
  kgen.iterate (I) in [(0), next, cond] {
    %0 = kgen.param.constant = <I>
    kgen.return %0 : index
  }
}

// -----

kgen.generator @get_list_element() {
  // expected-error @below {{custom op 'kgen.param.constant' expected a list type for 'get_list_element'}}
  %0 = kgen.param.constant = <get_list_element(:i32 0, 0)>
  kgen.return
}

// -----

kgen.generator @get_list_element() {
  // expected-error @below {{custom op 'kgen.param.constant' 'get_list_element' result should match list element type}}
  %0 = kgen.param.constant: i32 = <get_list_element(:list<index[1]> [0], 0)>
  kgen.return
}

// -----

kgen.generator @generatorWithTooManyConventions(
  %byval: !pop.pointer<i32>
  // expected-error @+1 {{too many parameter conventions specified, function has 1 value input}}
  ) conventions<none, byval, byref> {
  kgen.return
}

// -----

// expected-error @+1 {{argument #0 must have pointer type to have byref convention}}
kgen.func @bad(%byval: i32) conventions<none, byref> {
  kgen.return
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.func @callee(%byval: !pop.pointer<i32>) conventions<none, byref> {
  kgen.return
}

kgen.func @caller(%arg: !pop.pointer<i32>) {
  // Ok
  kgen.call @callee(%arg) conventions<byref> : (!pop.pointer<i32>) -> ()

  // expected-error @+1 {{symbol use conventions are array<i8: 0, 0> but @callee expected array<i8: 0, 1>}}
  kgen.call @callee(%arg) : (!pop.pointer<i32>) -> ()
  kgen.return
}

// -----

kgen.generator @target_params2<t0: target>()
 // expected-error @below {{expected '='}}
  constraints <[target_supports(:target t0, #kgen.target<triple"triple", "cpu", "features", 3, 4>), "must support target!!"]> {
  kgen.return
}
