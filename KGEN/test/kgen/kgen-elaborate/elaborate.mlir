// RUN: kgen-opt %s -split-input-file -elaborate-generators="enable-search=true allow-multiple-primary-impls=true" -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: kgen.func @parameter_use_chain()
kgen.generator @parameter_use_chain() {
  // Uses r2 and defines r1
  kgen.param.declare r1 = <add(r2, 1)>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <3>
  %0 = kgen.param.constant = <r1>

  // Uses 42 and defines r2
  kgen.param.declare r2 = <2>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <2>
  %1 = kgen.param.constant = <r2>

  // Uses r1/r2 and defines r3
  kgen.param.declare r3 = <mul(shl(r1, r2), 3)>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <36>
  %2 = kgen.param.constant = <r3>

  // Defines a dtype value and uses it.
  kgen.param.declare type1 : !kgen.dtype = <f32>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant: dtype = <f32>
  %3 = kgen.param.constant: !kgen.dtype = <type1>

  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL: @"unknown_attr,width=4"
kgen.generator @unknown_attr<width>() {
  // CHECK-NEXT: constant: simd<4, f32> = <*?>
  kgen.param.constant: simd<width, f32> = <*?>
  kgen.return
}

// CHECK-LABEL: @"empty_variadic,T=i32"
kgen.generator @empty_variadic<T: type>() {
  // CHECK-NEXT: constant: variadic<i32> = <[]>
  kgen.param.constant: variadic<T> = <[]>
  kgen.return
}

// CHECK-LABEL: @call_unknown_attr
kgen.generator @call_unknown_attr() {
  kgen.call @unknown_attr<4>() : () -> ()
  kgen.call @empty_variadic<:type i32>() : () -> ()
  kgen.return
}

// CHECK-NOT: kgen.generator @trivial_generator
// This gets "specialized" into a kernel.
kgen.generator @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}
// CHECK-LABEL: kgen.func @trivial_generator
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:    kgen.return %[[ARG0]] : si32
// CHECK-NEXT: }

kgen.generator @genA<size, DT: dtype, val: f32>(%arg0: si32) -> si32 {

  %0 = kgen.param.constant = <add(size, 4)>
  %1 = kgen.param.constant: dtype = <DT>
  %2 = kgen.param.constant: f32 = <val>

  // Silly op so we know when something used this.
  "genA.op"() { value = #kgen.param.decl.ref<"size"> : index} : () -> !pop.scalar<DT>

  kgen.return %arg0 : si32
}

// CHECK-LABEL: kgen.func @"genA,size=19,DT=si8,val=1.50{{.*}}"
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:    %[[V0:.*]] = kgen.param.constant  = <23>
// CHECK-NEXT:    %[[V1:.*]] = kgen.param.constant: dtype = <si8>
// CHECK-NEXT:    %[[V2:.*]] = kgen.param.constant: f32 = <1.500000e+00>
// CHECK-NEXT:    %[[V3:.*]] = "genA.op"() {value = 19 : index} : () -> !pop.scalar<si8>
// CHECK-NEXT:    kgen.return %[[ARG0]] : si32
// CHECK-NEXT:  }

// CHECK-LABEL: kgen.func @"genA,size=42,DT=f32,val=2.00{{.*}}"
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:   %[[V0:.*]] = kgen.param.constant  = <46>
// CHECK-NEXT:   %[[V1:.*]] = kgen.param.constant: dtype = <f32>
// CHECK-NEXT:   %[[V2:.*]] = kgen.param.constant: f32 = <2.000000e+00>
// CHECK-NEXT:   %[[V3:.*]] = "genA.op"() {value = 42 : index} : () -> !pop.scalar<f32>
// CHECK-NEXT:   kgen.return %[[ARG0]] : si32
// CHECK-NEXT: }

// CHECK-LABEL: kgen.func @call_generator_test
// CHECK-SAME: %[[ARG0:.*]]: si32, %[[ARG1:.*]]: si32
kgen.generator @call_generator_test(%arg0: si32, %arg1: si32) -> (si32, si32, si32) {
  // Can invoke the generator directly.
  %0 = kgen.call @trivial_generator(%arg0) : (si32) -> si32
  // CHECK-NEXT: %{{.*}} = kgen.call @trivial_generator(%[[ARG0]])

  // CHECK-NOT: kgen.param.declare
  kgen.param.declare our_size = <42>

  // Can invoke parameterized generators directly.
  // CHECK-NEXT: %{{.*}} = kgen.call @"genA,size=42,DT=f32,val=2.00{{.*}}"(%[[ARG0]]) : (si32) -> si32
  %1 = kgen.call @genA<our_size, :dtype f32, :f32 2.0>(%arg0) : (si32) -> si32

  // CHECK-NEXT: %{{.*}} = kgen.call @"genA,size=19,DT=si8,val=1.50{{.*}}"(%[[ARG1]]) : (si32) -> si32
  %2 = kgen.call @genA<19, :dtype si8, :f32 1.5>(%arg1) : (si32) -> si32

  // CHECK-NEXT: %{{.*}} = kgen.call @"genA,size=19,DT=si8,val=1.50{{.*}}"(%[[ARG1]]) : (si32) -> si32
  %3 = kgen.call @genA<19, :dtype si8, :f32 1.5>(%arg1) : (si32) -> si32

  kgen.return %0, %1, %2 : si32, si32, si32
}

// CHECK-LABEL: kgen.func @test_variadic_ptr_map
kgen.generator @test_variadic_ptr_map() {
  // CHECK-NEXT: %variadic = kgen.param.constant: variadic<type> = <[pointer<i32, 42>, pointer<f32, 42>, pointer<i8, 42>]>
  kgen.param.declare types : variadic<!kgen.type> = <[i32, f32, i8]>
  %variadic = kgen.param.constant: variadic<!kgen.type> = <variadic_ptr_map(:variadic<!kgen.type> types, 42)>
  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL: kgen.func @test_variadic_ptrremove_map
kgen.generator @test_variadic_ptrremove_map() {
  // CHECK-NEXT: %variadic = kgen.param.constant: variadic<type> = <[i32, f32, i8]>
  kgen.param.declare types : variadic<!kgen.type> = <[pointer<i32>, pointer<f32, 42>, pointer<i8>]>
  %variadic = kgen.param.constant: variadic<!kgen.type> = <variadic_ptrremove_map(:variadic<!kgen.type> types)>
  // CHECK-NEXT: kgen.return
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//

// Test that parameter and result argument types get rewritten and specialized.

// CHECK-LABEL: kgen.func @"float_constant_f32,value=1.50{{.*}},DT=f32"() -> !pop.scalar<f32> {
// ...
// CHECK:    %[[V1:.*]] = llvm.fptrunc
// CHECK:    %[[V2:.*]] = pop.cast_from_builtin %[[V1]] : f32 to !pop.scalar<f32>
// CHECK:    kgen.return %[[V2]] : !pop.scalar<f32>

kgen.generator @float_constant_f32<value: f64, DT: dtype>() -> !pop.scalar<DT> {
  kgen.param.assert <eq(:dtype DT, f32)>, "float please"
  %0 = kgen.param.constant: f64 = <value>
  %1 = llvm.fptrunc %0 : f64 to f32
  %2 = pop.cast_from_builtin %1: f32 to !pop.scalar<DT>
  kgen.return %2 : !pop.scalar<DT>
}

// CHECK-LABEL: kgen.func @test_f32() -> f32 {
// CHECK:    %[[V0:.*]] = kgen.call @"float_constant_f32,value=1.50{{.*}},DT=f32"() : () -> !pop.scalar<f32>
// CHECK:    %[[V1:.*]] = pop.cast_to_builtin %[[V0]] : !pop.scalar<f32> to f32
kgen.generator @test_f32() -> f32 {
  kgen.param.declare DT : dtype = <f32>
  %1 = kgen.call @float_constant_f32<:f64 1.5, :dtype DT>() : () -> !pop.scalar<DT>
  %2 = pop.cast_to_builtin %1 : !pop.scalar<DT> to f32
  kgen.return %2 : f32
}

// -----

//===----------------------------------------------------------------------===//

// Test that we can do static assertions on computed parameter expressions (e.g.
// those that are the result of a sub-generator invocation.


// CHECK-LABEL: kgen.func @paramAssertExample()
// CHECK-NOT: kgen.param.assert
kgen.generator @paramAssertExample() {
  kgen.param.declare flen = <4>

  // Should succeed.
  kgen.param.assert <eq(flen, 4)>, "vector length should be 4 for floats"
  kgen.return
}

// CHECK-LABEL: kgen.func @"parametricAdd,sz=1,dt=ui64"
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<ui64>, %[[ARG1:.*]]: !pop.scalar<ui64>) -> !pop.scalar<ui64> {
// CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !pop.scalar<ui64>
// CHECK-NEXT: kgen.return %[[V0]] : !pop.scalar<ui64>

// CHECK-LABEL: kgen.func @"parametricAdd,sz=2,dt=f32"
// CHECK-SAME: (%[[ARG0:.*]]: !pop.simd<2, f32>, %[[ARG1:.*]]: !pop.simd<2, f32>) -> !pop.simd<2, f32> {
// CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !pop.simd<2, f32>
// CHECK-NEXT: kgen.return %[[V0]] : !pop.simd<2, f32>

kgen.generator @parametricAdd<sz, dt: dtype>
  (%a: !pop.simd<sz,dt>, %b: !pop.simd<sz,dt>) -> !pop.simd<sz,dt> {
  %res = pop.add %a, %b : !pop.simd<sz,dt>
  kgen.return %res : !pop.simd<sz,dt>
}

// CHECK-LABEL: kgen.func @parametricTypes(
kgen.generator @parametricTypes(%arg0: !pop.scalar<ui64>, %arg1: !pop.simd<2, f32>) {
  kgen.param.declare dt: dtype = <ui32>
  kgen.param.declare ty1: type = <!pop.scalar<dt>>

  // CHECK-NEXT:   "impl.0"() : () -> !pop.scalar<ui32>
  "impl.0"() : () -> !kgen.paramref<ty1>

  // CHECK-NEXT: = kgen.call @"parametricAdd,sz=1,dt=ui64"
  // CHECK-SAME: (%[[ARG0:.*]], %[[ARG0:.*]]) : (!pop.scalar<ui64>, !pop.scalar<ui64>) -> !pop.scalar<ui64>
  %0 = kgen.call @parametricAdd<1, :dtype ui64>(%arg0, %arg0) : (!pop.scalar<ui64>, !pop.scalar<ui64>) -> !pop.scalar<ui64>

  // CHECK-NEXT: = kgen.call @"parametricAdd,sz=2,dt=f32"(%[[ARG1]], %[[ARG1]]) : (!pop.simd<2, f32>, !pop.simd<2, f32>) -> !pop.simd<2, f32>
  %1 = kgen.call @parametricAdd<2, :dtype f32>(%arg1, %arg1) : (!pop.simd<2, f32>, !pop.simd<2, f32>) -> !pop.simd<2, f32>

  kgen.return
}

// CHECK-LABEL: kgen.func @"takeUnary,dt=f32,fn=_nopExample"() {
// CHECK: %simd = kgen.param.constant
// CHECK: %0 = pop.cast %simd
// CHECK: %1 = kgen.call @"nopExample,dt=f32"(%0) : (!pop.scalar<f32>) -> !pop.scalar<f32>
// CHECK: %2 = kgen.call @"nopExample,dt=f32"(%1) : (!pop.scalar<f32>) -> !pop.scalar<f32>

// CHECK-LABEL: kgen.func @"takeUnary,dt=si32,fn=_doubleExample"()
// CHECK: %simd = kgen.param.constant
// CHECK: %0 = pop.cast %simd
// CHECK: %1 = kgen.call @"doubleExample,dt=si32"(%0) : (!pop.scalar<si32>) -> !pop.scalar<si32>
// CHECK: %2 = kgen.call @"doubleExample,dt=si32"(%1) : (!pop.scalar<si32>) -> !pop.scalar<si32>

kgen.generator @takeUnary
  <dt: dtype, fn: <dtype>(!pop.scalar<*(0,0)>) -> !pop.scalar<*(0,0)>>() {

  %one = kgen.param.constant: scalar<si64> = <1>
  %0 = pop.cast %one : !pop.scalar<si64> to !pop.scalar<dt>
  %1 = kgen.call_param[(!pop.scalar<dt>) -> !pop.scalar<dt>:
    bind_signature(:<dtype>(!pop.scalar<*(0,0)>) -> !pop.scalar<*(0,0)> fn, dt)](%0)
  %2 = kgen.call_param[(!pop.scalar<dt>) -> !pop.scalar<dt>:
    bind_signature(:<dtype>(!pop.scalar<*(0,0)>) -> !pop.scalar<*(0,0)> fn, dt)](%1)
  kgen.return
}

kgen.generator @doubleExample<dt:dtype>(%arg0: !pop.scalar<dt>) -> !pop.scalar<dt> {
  %0 = pop.add %arg0, %arg0: !pop.scalar<dt>
  kgen.return %0 : !pop.scalar<dt>
}

kgen.generator @nopExample<dt:dtype>(%arg0: !pop.scalar<dt>) -> !pop.scalar<dt> {
  kgen.return %arg0 : !pop.scalar<dt>
}

kgen.generator @takeParametricBinary
  <sz,
   dt: dtype,
   fn: <index, dtype>(!pop.simd<*(0,0),*(0,1)>, !pop.simd<*(0,0),*(0,1)>) -> !pop.simd<*(0,0),*(0,1)>
  >() {

  %one = kgen.param.constant: scalar<si64> = <1>
  %0 = pop.cast %one : !pop.scalar<si64> to !pop.scalar<dt>

  %1 = kgen.call_param[(!pop.scalar<dt>, !pop.scalar<dt>) -> !pop.scalar<dt>:
    bind_signature(:<index, dtype>(!pop.simd<*(0,0),*(0,1)>, !pop.simd<*(0,0),*(0,1)>) -> !pop.simd<*(0,0),*(0,1)> fn, 1, dt)](%0, %0)
  kgen.return
}

// CHECK-LABEL:  kgen.func @test_symbol() {
kgen.generator @test_symbol() {
  // CHECK: kgen.call @"takeUnary,dt=si32,fn=_doubleExample"()
  kgen.call @takeUnary<:dtype si32,
     :<dtype>(!pop.scalar<*(0,0)>) -> !pop.scalar<*(0,0)> @doubleExample>() : () -> ()

  // CHECK: kgen.call @"takeUnary,dt=f32,fn=_nopExample"()
  kgen.call @takeUnary<:dtype f32,
     :<dtype>(!pop.scalar<*(0,0)>) -> !pop.scalar<*(0,0)> @nopExample>() : () -> ()

  // CHECK: kgen.call @"takeParametricBinary,sz=2,dt=f32,fn=_parametricAdd"()
  kgen.call @takeParametricBinary
     <
      2,
      :dtype f32,
      :<index, dtype>(!pop.simd<*(0,0), *(0,1)>, !pop.simd<*(0,0), *(0,1)>) -> !pop.simd<*(0,0), *(0,1)> @parametricAdd
     >() : () -> ()

  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @"parametricBinOp,ty=scalar<f32>"
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>, %[[ARG1:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
// CHECK-NEXT: %[[V0:.*]] = "custom.op"(%[[ARG0]], %[[ARG1]]) : (!pop.scalar<f32>, !pop.scalar<f32>) -> !pop.scalar<f32>
// CHECK-NEXT: kgen.return %[[V0]] : !pop.scalar<f32>
kgen.generator @parametricBinOp<ty: type>
  (%a: !kgen.paramref<ty>, %b: !kgen.paramref<ty>) -> !kgen.paramref<ty> {
  %res = "custom.op" (%a, %b) : (!kgen.paramref<ty>, !kgen.paramref<ty>) -> !kgen.paramref<ty>
  kgen.return %res : !kgen.paramref<ty>
}

// CHECK-LABEL: kgen.func @"takeParametricBinary,dt=f32,fn=_parametricBinOp"() {
kgen.generator @takeParametricBinary
  <dt: dtype,
   fn: <type>(!kgen.paramref<*(0,0)>, !kgen.paramref<*(0,0)>) -> !kgen.paramref<*(0,0)>
  >() {

  %one = kgen.param.constant: scalar<si64> = <1>
  %0 = pop.cast %one : !pop.scalar<si64> to !pop.scalar<dt>

  // CHECK: kgen.call @"parametricBinOp,ty=scalar<f32>"
  %1 = kgen.call_param[(!pop.scalar<dt>, !pop.scalar<dt>) -> !pop.scalar<dt>:
    bind_signature(:<type>(!kgen.paramref<*(0,0)>, !kgen.paramref<*(0,0)>) -> !kgen.paramref<*(0,0)>
      fn, !pop.scalar<dt>)](%0, %0)
  kgen.return
}

// CHECK-LABEL: kgen.func @test_paramref_type_rewrite() {
kgen.generator @test_paramref_type_rewrite() {
  // CHECK: kgen.call @"takeParametricBinary,dt=f32,fn=_parametricBinOp"() : () -> ()
  kgen.call @takeParametricBinary<:dtype f32,
      :<type>(!kgen.paramref<*(0,0)>, !kgen.paramref<*(0,0)>) -> !kgen.paramref<*(0,0)> @parametricBinOp>() : () -> ()

  kgen.return
}


kgen.generator @indirect_callee() -> index {
  %0 = kgen.param.constant = <42>
  kgen.return %0: index
}

kgen.generator @call_indirect(%fp: !kgen.signature<() -> index>) -> index {
  %0 = kgen.call_indirect %fp() : () -> index
  kgen.return %0: index
}

// CHECK-LABEL: kgen.func @test_comptime_call_indirect
kgen.generator @test_comptime_call_indirect() -> index {
  // CHECK-NEXT: %index42 = kgen.param.constant = <42>
  %0 = kgen.param.constant = <apply(:(!kgen.signature<() -> index>) -> index @call_indirect, @indirect_callee)>
  kgen.return %0: index
}

// -----

// This takes a parameter function that uses a contextual type instead of
// to-be-bound types.
// CHECK-LABEL: kgen.func @"takeFnContextualType,ty=index,fn=_sillyFn"() -> index {
// CHECK:  %0 = kgen.call @sillyFn() : () -> index
kgen.generator @takeFnContextualType<ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> {
  %0 = kgen.call_param[()->!kgen.paramref<ty>: fn]()
  kgen.return %0: !kgen.paramref<ty>
}

kgen.generator @sillyFn() -> index {
  %0 = kgen.param.constant = <42>
  kgen.return %0: index
}

// CHECK-LABEL:  kgen.func @elaborateFnWithContextualType() -> index {
// CHECK:   %0 = kgen.call @"takeFnContextualType,ty=index,fn=_sillyFn"() : () -> index
kgen.generator @elaborateFnWithContextualType() -> index {
  %0 = kgen.call @takeFnContextualType<:type index, :()->index @sillyFn>() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: @elaborateFnWithContextualType2()
kgen.generator @elaborateFnWithContextualType2() -> (index, index) {
  // Show we can bind a generic signature to a concrete one.
  kgen.param.declare boundFn: ()->index =
    <bind_signature(:<type, ()->!kgen.paramref<*(1,0)>>() -> !kgen.paramref<*(0,0)> @takeFnContextualType,
                    index, @sillyFn)>

  // CHECK-NEXT: %0 = kgen.call @"takeFnContextualType,ty=index,fn=_sillyFn"()
  %0 = kgen.call_param[()->index: boundFn]()

  kgen.param.declare fn: <type, ()->!kgen.paramref<*(1,0)>>() -> !kgen.paramref<*(0,0)> = <@takeFnContextualType>

  kgen.param.declare boundFn2: ()->index =
    <bind_signature(:<type, ()->!kgen.paramref<*(1,0)>>() -> !kgen.paramref<*(0,0)> fn,
                    index, @sillyFn)>

  // CHECK-NEXT: %1 = kgen.call @"takeFnContextualType,ty=index,fn=_sillyFn"()
  %1 = kgen.call_param[()->index: boundFn2]()

  kgen.return %0, %1 : index, index
}

// -----

// CHECK-LABEL: kgen.func @"takeStringParameter,SomeString=\22foo\22"
kgen.generator @takeStringParameter<SomeString: string>() {
  kgen.param.assert <eq(:string SomeString, "foo")>, "I want foo"
  kgen.return
}

// CHECK-LABEL: kgen.func @giveString
kgen.generator @giveString() {
  // CHECK-NEXT: kgen.call @"takeStringParameter,SomeString=\22foo\22"
  kgen.call @takeStringParameter<:string "foo">() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: @"makeListConst,A=1"
kgen.generator @makeListConst<A>() {
  // CHECK-NEXT: kgen.param.constant: array<2, index> = <[1, 1]>
  %0 = kgen.param.constant: array<2, index> = <[A, A]>
  kgen.return
}

kgen.generator @doIt() {
  kgen.call @makeListConst<1>() : () -> ()
  kgen.return
}

// CHECK-LABEL: @"variableList,N=2,Ts=[1, 2]"
kgen.generator @variableList<N, Ts: array<N, i32>>() {
  // CHECK-NEXT: kgen.param.constant: array<2, i32> = <[1, 2]>
  %0 = kgen.param.constant: array<N, i32> = <Ts>
  kgen.return
}

kgen.generator @passTypeList() {
  kgen.call @variableList<2, :array<2, i32> [1, 2]>() : () -> ()
  kgen.return
}

// CHECK-LABEL: @"type_of_unknown
kgen.generator @type_of_unknown<T: type, value: !kgen.paramref<T>>() {
  // CHECK-NEXT: <0>
  kgen.param.constant: i1 = <eq(:!kgen.paramref<T> value, *?)>
  kgen.return
}

kgen.generator @check() {
  kgen.call @type_of_unknown<:type i32, :i32 1>() : () -> ()
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Recursion Test
//===----------------------------------------------------------------------===//
//
// This shows that we properly support recursive expansion.
//

kgen.generator @genItf3<x>() {
  kgen.param.if <eq(x, 0)> {
    "impl.0"() {attr=#kgen.param.decl.ref<"x"> : index}: () -> ()
    kgen.param.yield
  } else {
    "impl.1"() {attr=#kgen.param.decl.ref<"x"> : index} : () -> ()
    kgen.call @genItf3<sub(x, 1)>() : () -> ()
    kgen.param.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @"genItf3,x=0"()
// CHECK-NEXT:   "impl.0"() {attr = 0 : index}

// CHECK-LABEL: kgen.func @"genItf3,x=1"()
// CHECK-NEXT:   "impl.1"() {attr = 1 : index}
// CHECK-NEXT:   kgen.call @"genItf3,x=0"()

// CHECK-LABEL: kgen.func @"genItf3,x=2"()
// CHECK-NEXT:   "impl.1"() {attr = 2 : index}
// CHECK-NEXT:   kgen.call @"genItf3,x=1"()

// CHECK-LABEL: kgen.func @"genItf3,x=3"()
// CHECK-NEXT:   "impl.1"() {attr = 3 : index}
// CHECK-NEXT:   kgen.call @"genItf3,x=2"()

// CHECK-LABEL: kgen.func @"genItf3,x=4"()
// CHECK-NEXT:   "impl.1"() {attr = 4 : index}
// CHECK-NEXT:   kgen.call @"genItf3,x=3"()

// CHECK-LABEL:   kgen.func @use_Itf3() {
// CHECK-NEXT:      kgen.call @"genItf3,x=4"() : () -> ()
// CHECK-NEXT:      kgen.call @"genItf3,x=2"() : () -> ()
// CHECK-NEXT:      kgen.return
kgen.generator @use_Itf3() {
  kgen.call @genItf3<4>() : () -> ()
  kgen.call @genItf3<2>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @"param_add,A=1,B=2"
kgen.generator @param_add<A, B>() -> index {
  // CHECK-NEXT: %index3 = kgen.param.constant = <3>
  %0 = kgen.param.constant = <add(A, B)>
  // CHECK-NEXT: kgen.return %index3
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @partial_bind_signature_region
kgen.generator @partial_bind_signature_region() -> index {
  kgen.param.declare BoundFn: <index>() -> index = <bind_signature(:<index, index>() -> index @param_add, 1, ?)>
  // CHECK-NEXT: %0 = kgen.call @"param_add,A=1,B=2"() : () -> index
  %0 = kgen.call_param[() -> index: bind_signature(:<index>() -> index BoundFn, 2)]()
  // CHECK-NEXT: kgen.return %0
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @"param_add2,A=1,B=2,C=3"
kgen.generator @param_add2<A, B, C>() -> index {
  // CHECK-NEXT: %index4 = kgen.param.constant = <4>
  %0 = kgen.param.constant = <add(sub(B, A), C)>
  // CHECK-NEXT: kgen.return %index4
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @partial_bind_signature_region_2
kgen.generator @partial_bind_signature_region_2() -> index {
  kgen.param.declare BoundFn: <index, index>() -> index = <bind_signature(:<index, index, index>() -> index @param_add2, 1, ?, ?)>
  kgen.param.declare BoundFn2: <index>() -> index = <bind_signature(:<index, index>() -> index BoundFn, ?, 3)>
  // CHECK-NEXT: %0 = kgen.call @"param_add2,A=1,B=2,C=3"() : () -> index
  %0 = kgen.call_param[() -> index: bind_signature(:<index>() -> index BoundFn2, 2)]()
  // CHECK-NEXT: kgen.return %0
  kgen.return %0 : index
}

// -----

// COM: First instantiation of `@fwd` is inside an assert.

kgen.generator @fwd(%a: i1) -> i1 {
  kgen.return %a : i1
}

kgen.generator @f() {
  kgen.param.assert <apply(:(i1) -> i1 @fwd, 1)>, "true"
  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // CHECK-NEXT: call @f
  kgen.call @f() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: @"g1,size=3"
// CHECK-LABEL: @"g1,size=5"
kgen.generator @g1<size>() -> index {
  %0 = kgen.param.constant = <size>
  kgen.return %0 : index
}

// CHECK-LABEL: @"g2,size=3,width=5"
kgen.generator @g2<size, width>() -> index {
  // CHECK-NEXT: call @"g1,size=5"
  %0 = kgen.call @g1<width>() : () -> index
  // CHECK-NEXT: call @"g1,size=3"
  %1 = kgen.call @g1<size>() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: @root
kgen.generator @root() {
  kgen.param.declare q = <3>
  kgen.param.declare w = <5>
  // CHECK-NEXT: kgen.call @"g2,size=3,width=5"
  %0 = kgen.call @g2<q, w>() : () -> index
  kgen.return
}

// -----

// COM: Check that `elaborate-generators` attaches the host target info.

// CHECK: module attributes {M.target_info = #M.target<{{.*}}>}

kgen.generator @some_func() {
  kgen.return
}

// -----

// CHECK-LABEL: @constexprIfNoParams()
kgen.generator @constexprIfNoParams() {
  // CHECK-NEXT: "should.appear"
  kgen.param.if<1> {
    "should.appear"() : () -> ()
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL: @constexprIfBasic()
kgen.generator @constexprIfBasic() {
  kgen.param.declare cond_var = <32>

  // CHECK-NEXT: "should.appear"
  %0 = kgen.param.if <lt(cond_var, 10)> -> index {
    %1 = "should.not.appear"() : () -> index
    kgen.param.declare next_lt = <add(cond_var, 10)>
    kgen.param.yield %1 : index
  } else {
    %3 = "should.appear"() : () -> index
    kgen.param.declare next_gt = <add(cond_var, 20)>
    kgen.param.yield %3 : index
  }

  kgen.return
}

// CHECK-LABEL: @nestedConstexprIf()
kgen.generator @nestedConstexprIf() {
  kgen.param.declare cond_var = <32>

  // CHECK-NEXT: "should.appear"
  // CHECK-NOT: "should.not.appear"
  %0 = kgen.param.if <lt(cond_var, 10)> -> index {
    %1 = "should.not.appear"() : () -> index
    kgen.param.declare next_lt = <add(cond_var, 10)>
    kgen.param.yield %1 : index
  } else {
    %3 = kgen.param.if <gt(cond_var, 30)> -> index {
      %4 = "should.appear"() : () -> index
      kgen.param.yield %4 : index
    } else {
      %4 = "should.not.appear"() : () -> index
      kgen.param.yield %4 : index
    }
    kgen.param.yield %3 : index
  }

  kgen.return
}

// CHECK-LABEL: @nestedConstexprIf2()
kgen.generator @nestedConstexprIf2() {
  kgen.param.declare cond_var = <32>

  %0 = kgen.param.if <lt(cond_var, 10)> -> index {
    %1 = "should.not.appear"() : () -> index
    kgen.param.declare next_lt = <add(cond_var, 10)>
    kgen.param.yield %1 : index
  } else {
    // CHECK-NEXT: param.constant: i1 = <1>
    %condition = kgen.param.constant : i1 = <gt(cond_var, 30)>
    // CHECK-NEXT: hlcf.if
    %3 = hlcf.if %condition -> index {
      // CHECK-NEXT: "should.appear"
      %4 = "should.appear"() : () -> index
      kgen.param.declare next_inner = <35>
      // CHECK-NEXT: hlcf.yield
      hlcf.yield %4 : index
      // CHECK-NEXT: else
    } else {
      // CHECK-NEXT: "should.also.appear"
      %4 = "should.also.appear"() : () -> index
      // CHECK-NEXT: hlcf.yield
      hlcf.yield %4 : index
    }
    // CHECK-NOT: param.yield
    kgen.param.yield %3 : index
  }

  kgen.return
}

// -----

// CHECK-LABEL: @"constexprIfInputParam,x=11"
kgen.generator @constexprIfInputParam<x>() {
  // CHECK-NEXT: "should.appear"
  %0 = kgen.param.if <gt(x, 10)> -> index {
    %1 = "should.appear"() : () -> index
    kgen.param.declare next_lt = <add(x, 10)>
    kgen.param.yield %1 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.declare next_gt = <add(x, 20)>
    kgen.param.yield %3 : index
  }

  kgen.return
}

kgen.generator @caller() {
  kgen.call @constexprIfInputParam<11>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: @constexprIfEarlyExit
kgen.generator @constexprIfEarlyExit() -> index {
  kgen.param.declare x = <11>
  // CHECK-NEXT: [[RES:%[0-9]+]] = "should.appear"
  %0 = kgen.param.if <gt(x, 10)> -> index {
    %1 = "should.appear"() : () -> index
    // CHECK-NEXT: kgen.return [[RES]]
    kgen.return %1 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.yield %3 : index
  }
  // CHECK-NOT: param.constant = <32>
  %4 = kgen.param.constant = <32>

  kgen.return %0 : index
}

// COM: This ensures that the blocks after the early exit are correctly
// COM: removed *without* a use-after-free during elaboration.
// CHECK-LABEL: @constexprIfEarlyExit2
kgen.generator @constexprIfEarlyExit2() -> index {
  kgen.param.declare x = <11>
  // CHECK-NEXT: [[RES:%[0-9]+]] = "should.appear"
  %0 = kgen.param.if <gt(x, 10)> -> index {
    %1 = "should.appear"() : () -> index
    // CHECK-NEXT: kgen.return [[RES]]
    kgen.return %1 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.yield %3 : index
  }
  // CHECK-NOT: "should.not.appear"
  kgen.param.if <gt(x, 10)> {
    "should.not.appear"() : () -> ()
    %4 = index.constant 3
    // CHECK-NOT: kgen.return
    kgen.return %4 : index
  } else {
    kgen.param.yield
  }
  // CHECK-NOT: param.constant = <32>
  %4 = kgen.param.constant = <32>

  kgen.return %0 : index
}

// CHECK-LABEL: @constexprIfEarlyExitWithParam
kgen.generator @constexprIfEarlyExitWithParam() -> index {
  // CHECK-NEXT: [[RES:%[0-9]+]] = "should.appear"
  %0 = kgen.param.if <gt(x, 10)> -> index {
    %1 = "should.appear"() : () -> index
    // CHECK-NEXT: kgen.return [[RES]]
    kgen.return %1 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.yield %3 : index
  }
  kgen.param.declare x = <11>
  // CHECK-NOT: param.constant = <32>
  %4 = kgen.param.constant = <32>

  kgen.return %0 : index
}

// CHECK-LABEL: @constexprIfEarlyExitWithParam2
kgen.generator @constexprIfEarlyExitWithParam2() -> index {
  // CHECK-NEXT: param.constant = <11>
  %0 = kgen.param.constant = <x>
  // CHECK-NEXT: [[RES:%[0-9]+]] = "should.appear"
  %1 = kgen.param.if <1> -> index {
    %2 = "should.appear"() : () -> index
    // CHECK-NEXT: kgen.return [[RES]]
    kgen.return %2 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.yield %3 : index
  }
  kgen.param.declare x = <11>
  // CHECK-NOT: param.constant = <32>
  %4 = kgen.param.constant = <32>

  kgen.return %1 : index
}

// -----

kgen.generator @returnTrue() -> i1 {
  %0 = kgen.param.constant: i1 = <1>
  kgen.return %0 : i1
}

// CHECK-LABEL: @constexprIfFunctionCallCondition
kgen.generator @constexprIfFunctionCallCondition() -> index {
  // CHECK-NEXT: [[RES:%[0-9]+]] = "should.appear"
  %1 = kgen.param.if <apply(:() -> i1 @returnTrue)> -> index {
    %2 = "should.appear"() : () -> index
    // CHECK-NEXT: kgen.return [[RES]]
    kgen.param.yield %2 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.yield %3 : index
  }

  kgen.return %1 : index
}

kgen.generator @returnInputParam(%arg0: !kgen.struct<(scalar<bool>)>) -> i1 {
  %1 = kgen.struct.extract %arg0[0] : !kgen.struct<(scalar<bool>)>
  %2 = pop.cast_to_builtin %1: !pop.scalar<bool> to i1
  kgen.return %2 : i1
}

kgen.generator @returnTrueStruct() -> !kgen.struct<(scalar<bool>)> {
  %0 = kgen.param.constant: scalar<bool> = <<true>>
  %1 = kgen.struct.create(%0) : !kgen.struct<(scalar<bool>)>
  kgen.return %1 : !kgen.struct<(scalar<bool>)>
}

// CHECK-LABEL: @"ifFn
kgen.generator @ifFn<true: !kgen.struct<(scalar<bool>)>>() -> index {
  // CHECK-NEXT: [[RES:%[0-9]+]] = "should.appear"
  %1 = kgen.param.if <apply(:(!kgen.struct<(scalar<bool>)>) -> i1 @returnInputParam, true)> -> index {
    %2 = "should.appear"() : () -> index
    // CHECK-NEXT: kgen.return [[RES]]
    kgen.param.yield %2 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.yield %3 : index
  }

  kgen.return %1 : index
}

// CHECK-LABEL: @constexprIfFunctionCallCondition2
kgen.generator @constexprIfFunctionCallCondition2() {
  kgen.param.declare true: !kgen.struct<(scalar<bool>)> = <apply(:() -> !kgen.struct<(scalar<bool>)> @returnTrueStruct)>
  %0 = kgen.call @ifFn<:!kgen.struct<(scalar<bool>)> true>() : () -> index
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @substitute_current_target
kgen.generator @substitute_current_target() {
  // CHECK-NEXT: constant: target = <#kgen.target<triple = {{.*}}>>
  kgen.param.constant: target = <current_target()>
  kgen.return
}

// -----

// CHECK: module
// CHECK-NOT: kgen.func

kgen.generator @not_a_primary_generator<N>() {
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @rebind_parameter
kgen.generator @rebind_parameter() {
  // CHECK-NEXT: constant: array<2, index> = <[1, 2]>
  kgen.param.declare size = <2>
  kgen.param.declare list_input: array<2, index> = <[1, 2]>
  kgen.param.declare list_output: array<size, index> = <rebind(:array<2, index> list_input)>
  kgen.param.constant: array<size, index> = <list_output>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @recurse
// CHECK-SAME: () {
// CHECK-NEXT:  kgen.call @recurse() : () -> ()
// CHECK-NEXT:  kgen.return
// CHECK-NEXT:  }
kgen.generator @recurse() {
  kgen.call @recurse() : () -> ()
  kgen.return
}

// -----

// COM: Tricky recursion order.

// CHECK-LABEL: kgen.func @err
kgen.generator @err() {
  // CHECK-NEXT: call @call
  kgen.call @call() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() {
  // CHECK-NEXT: call @getattr
  kgen.call @getattr() : () -> ()
  // CHECK-NEXT: call @call
  kgen.call @call() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @getattr
kgen.generator @getattr() {
  // CHECK-NEXT: call @err
  kgen.call @err() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @call
kgen.generator @call() {
  // CHECK-NEXT: call @err
  kgen.call @err() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @unpack_in_type
kgen.generator @unpack_in_type() {
  // CHECK-NEXT: array<1, index>
  %0 = pop.stack_allocation 1 x array<apply(:() -> index @produce_one), index>
  kgen.return
}

kgen.generator @produce_one() -> index {
  %0 = kgen.param.constant: index = <1>
  kgen.return %0 : index
}

// CHECK-LABEL: func @"paramRecurse,in=0"()
// CHECK-NEXT: return

// CHECK-LABEL: func @"paramRecurse,in=1"()
// CHECK-NEXT: call @"paramRecurse,in=0"

// CHECK-LABEL: func @"paramRecurse,in=2"()
// CHECK-NEXT: call @"paramRecurse,in=1"

// CHECK-LABEL:func  @"paramRecurse,in=3"()
// CHECK-NEXT: call @"paramRecurse,in=2"

kgen.generator @paramRecurse<in>() {
  kgen.param.if <eq(in, 0)> {
    kgen.param.yield
  } else {
    kgen.call @paramRecurse<add(in, -1)>() : () -> ()
    kgen.param.yield
  }
  kgen.return
}

kgen.generator @caller() {
  kgen.call @paramRecurse<3>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @pointer_attr_elaborate
kgen.generator @pointer_attr_elaborate() {
  // CHECK-NEXT: kgen.param.constant: pointer<i8> = <0>
  kgen.param.declare type1: type = <i8>
  %0 = kgen.param.constant: pointer<type1> = <0>
  kgen.return
}

// -----

// COM: https://github.com/modularml/modular/issues/9745

// CHECK-LABEL: kgen.func @true_inside_false_param_if
kgen.generator @true_inside_false_param_if() {
  // CHECK-NEXT: should.appear
  // CHECK-NEXT: kgen.return
  kgen.param.if <0> {
    "should.not.appear"() : () -> ()
    kgen.return
  } else {
    kgen.param.if <1> {
      "should.appear"() : () -> ()
      kgen.return
    } else {
      "should.not.appear"() : () -> ()
      kgen.param.yield
    }
    kgen.param.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @"param_if_different,cond=0"
// CHECK-NEXT: constant = <3>

// CHECK-LABEL: kgen.func @"param_if_different,cond=1"
// CHECK-NEXT: constant = <2>
kgen.generator @param_if_different<cond: i1>() {
  kgen.param.declare a = <3>
  kgen.param.if <cond> {
    kgen.param.declare b = <2>
    kgen.param.constant = <b>
    kgen.param.yield
  } else {
    kgen.param.constant = <a>
    kgen.param.yield
  }
  kgen.return
}

kgen.generator @instantiate() {
  kgen.call @param_if_different<:i1 1>() : () -> ()
  kgen.call @param_if_different<:i1 0>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @async_function()
kgen.generator @async_function() async {
  kgen.return
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() {
  // CHECK: co.invoke[() async -> (): @async_function]
  kgen.param.declare fn: () async -> () = <@async_function>
  co.invoke[() async -> (): fn]()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @async_fn() async
kgen.generator @async_fn() async {
  kgen.return
}

// CHECK-LABEL: kgen.func export @nonparametric_async_call
kgen.generator export @nonparametric_async_call() {
  // CHECK-NEXT: co.invoke[() async -> (): @async_fn]
  co.invoke[() async -> (): @async_fn]()
  kgen.return
}

// -----

// COM: Check conditional parameter expressions.

kgen.generator @add_param<a : index>(%v : index) -> index {
  %0 = kgen.param.constant : index = <a>
  %1 = index.add %v, %0
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func @"add_param,a=1"
// CHECK-NOT: kgen.func @"add_param,a=2"
// CHECK-NOT: kgen.func @"add_param,a=3"
// CHECK: kgen.func @"add_param,a=4"
// CHECK-LABEL: kgen.func @param_cond
kgen.generator @param_cond() -> () {
  kgen.param.declare cond_false : i1 = <0>
  kgen.param.declare cond_true : i1 = <1>

  // COM: This should NOT evaluate @add_param<2> during parameter evaluation
  %5 = kgen.param.constant: index = <cond(cond_true,
        apply(:(index) -> index @add_param<1>, 0), apply(:(index) -> index @add_param<2>, 0))>
  // COM: This should NOT evaluate @add_param<3> during parameter evaluation
  %6 = kgen.param.constant: index = <cond(cond_false,
        apply(:(index) -> index @add_param<3>, 0), apply(:(index) -> index @add_param<4>, 0))>

  kgen.return
}

// -----

// CHECK: kgen.func @"callee,a=1"
// CHECK: kgen.func @"callee,a=2"

kgen.generator @callee<a>(%arg0: index) {
  kgen.return
}

kgen.generator @entry(%arg0: index) {
  // CHECK: create_closure[(index) -> (): @"callee,a=1"]
  kgen.create_closure[(index) -> (): @callee<1>]()
  kgen.param.declare fn: (index) -> () = <@callee<2>>
  // CHECK: create_closure[(index) -> (): @"callee,a=2"]
  kgen.create_closure[(index) -> (): fn](%arg0)
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @"recurse,axis=0"
// CHECK-NEXT:    kgen.return %arg0 : index

// CHECK-LABEL: kgen.func @"recurse,axis=1"
// CHECK-NEXT:    %0 = kgen.call @"recurse,axis=0"(%arg0) : (index) -> index
// CHECK-NEXT:    kgen.return %0 : index

// CHECK-LABEL: kgen.func @"recurse,axis=2"
// CHECK-NEXT:    %0 = kgen.call @"recurse,axis=1"(%arg0) : (index) -> index
// CHECK-NEXT:    kgen.return %0 : index

kgen.generator @recurse<axis>(%arg0: index) -> index {
  kgen.param.if <eq(axis, 0)> {
    kgen.return %arg0 : index
  } else {
    kgen.param.yield
  }
  %0 = kgen.call @recurse<add(axis, -1)>(%arg0) : (index) -> index
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @main
kgen.generator @main() {
  %idx42 = index.constant 42
  // CHECK: kgen.call @"recurse,axis=2"(%idx42) : (index) -> index
  %1 = kgen.call @recurse<2>(%idx42) : (index) -> index
  kgen.return
}

// -----

kgen.generator @take_closure(%arg0: !kgen.signature<(index) capturing -> index>, %arg1: index) {
  %0 = kgen.call_indirect %arg0(%arg1) : (index) capturing -> index
  kgen.return
}

// COM: Ensure that regions lifted by OutlineClosures pass are not erased
// CHECK-LABEL: kgen.func @"foo_k,N=5,M=3"() capturing -> !pop.scalar<index> {
kgen.generator @foo_k<N, M>() capturing -> !pop.scalar<index> {
  %0 = kgen.param.constant: scalar<index> = <0>
  kgen.return %0 : !pop.scalar<index>
}

// CHECK-LABEL: kgen.func @"foo,N=5"(%arg0: !pop.scalar<index>) {
kgen.generator @foo<N>(%arg0: !pop.scalar<index>) {
  kgen.param.declare k: <index>() capturing -> !pop.scalar<index> = <@foo_k<N, ?>>
  // CHECK: kgen.create_closure[() capturing -> !pop.scalar<index>: @"foo_k,N=5,M=3"]()
  %1 = kgen.create_closure[() capturing -> !pop.scalar<index>: bind_signature(:<index>() capturing -> !pop.scalar<index> k, 3)]()
  kgen.return
}

// CHECK-LABEL: kgen.func @main
kgen.generator @main() {
  %simd = kgen.param.constant: scalar<index> = <0>
  kgen.param.declare Bound: (!pop.scalar<index>) -> () = <@foo<5>>
  // CHECK: kgen.call @"foo,N=5"(%simd) : (!pop.scalar<index>) -> ()
  kgen.call_param[(!pop.scalar<index>) -> (): Bound](%simd)
  kgen.return
}

// COM: Ensure that staged closures follow the global store
kgen.generator @take_bat(%arg0: !kgen.signature<(index) capturing -> index>) {
	kgen.return
}

kgen.generator @bat(%arg0: index) capturing -> index {
	kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.func @bat_binder
kgen.generator @bat_binder(%arg0: index) {
  // CHECK: kgen.create_closure[(index) capturing -> index: @bat]()
	%2 = kgen.create_closure[(index) capturing -> index: h]()
	kgen.param.declare h: (index) capturing -> index = <@bat>
	kgen.return
}

// -----

kgen.generator @count_ops(%arg0: i1) -> index {
  %0 = hlcf.if %arg0 -> index {
    %idx0 = index.constant 0
    hlcf.yield %idx0 : index
  } else {
    %idx1 = index.constant 1
    hlcf.yield %idx1 : index
  }
  kgen.return %0 : index
}

kgen.generator @cost_of<fn: (i1) -> index>() -> index {
  %0:8 = kgen.cost_of[(i1) -> index: fn]
  // CHECK: %loads, %stores, %additions, %comparisons, %divisions, %multiplications, %multiplyAdds, %other = kgen.cost_of[(i1) -> index: @count_ops]
  kgen.return %0#7  : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() {
  // CHECK-NEXT: <1>
  %0 = kgen.param.constant = <apply(:() -> index @cost_of<:(i1) -> index @count_ops>)>
  kgen.return
}

// -----

kgen.generator @pop_add(%arg1: !pop.scalar<si32>) -> !pop.scalar<si32> {
  %0 = pop.add %arg1, %arg1 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

kgen.generator @cost_of_pop_add<fn: (!pop.scalar<si32>) -> !pop.scalar<si32>>() -> index {
  // CHECK: %loads, %stores, %additions, %comparisons, %divisions, %multiplications, %multiplyAdds, %other = kgen.cost_of[(!pop.scalar<si32>) -> !pop.scalar<si32>: @pop_add]
  %0:8 = kgen.cost_of[(!pop.scalar<si32>) -> !pop.scalar<si32>: fn]
  kgen.return %0#2  : index
}

kgen.generator @index_add(%arg1: index) -> index {
  // CHECK: %loads, %stores, %additions, %comparisons, %divisions, %multiplications, %multiplyAdds, %other = kgen.cost_of[(index) -> index: @index_add]
  %0 = index.add %arg1, %arg1
  kgen.return %0 : index
}

kgen.generator @cost_of_index_add<fn: (index) -> index>() -> index {
  %0:8 = kgen.cost_of[(index) -> index: fn]
  kgen.return %0#2  : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() {
  // CHECK-NEXT: <1>
  %0 = kgen.param.constant = <apply(:() -> index @cost_of_pop_add<:(!pop.scalar<si32>) -> !pop.scalar<si32> @pop_add>)>
  // CHECK-NEXT: <1>
  %1 = kgen.param.constant = <apply(:() -> index @cost_of_index_add<:(index) -> index @index_add>)>
 kgen.return
}

// -----

// CHECK: kgen.global @global_var : f32 [@global_init, @global_dtor](0)
kgen.global @global_var : f32 [@global_init, @global_dtor](0)

// CHECK: kgen.func @global_init
kgen.generator @global_init() {
  kgen.return
}

// CHECK: kgen.func @global_dtor
kgen.generator @global_dtor() {
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @preelaborated()
kgen.func @preelaborated() {
  // CHECK-NEXT: kgen.return
  kgen.return
}

// -----

module attributes {kgen.env = #kgen.env<{unit_value, int_value = 42 : index, str_value = "hello" : !kgen.string}>} {
  // CHECK-LABEL: kgen.func @env_test
  kgen.generator @env_test() {
    // CHECK-NEXT: i1 = <0>
    kgen.param.constant: i1 = <get_env("doesnt_exist")>
    // CHECK-NEXT: i1 = <1>
    kgen.param.constant: i1 = <get_env("unit_value")>
    // CHECK-NEXT: <42>
    kgen.param.constant = <get_env("int_value")>
    // CHECK-NEXT: string = <"hello">
    kgen.param.constant: string = <get_env("str_value")>
    kgen.return
  }
}

// -----

kgen.func @already_concrete() -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func export @interpret_concrete
kgen.generator export @interpret_concrete() {
  // CHECK-NEXT: = <0>
  kgen.param.constant = <apply(:() -> index @already_concrete)>
  kgen.return
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="foobar", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @custom_target
  kgen.generator @custom_target() {
    // CHECK: constant: i1 = <1>
    kgen.param.constant: i1 = <target_has_feature(current_target(), "foobar")>
    kgen.return
  }
}

// -----

kgen.generator @kernel() {
  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // COM: Just check that the code compiles. The assembly is target-dependent.
  // CHECK: constant: struct
  kgen.param.constant: struct<(string, index, (!kgen.pointer<pointer<none>>) capturing -> !kgen.none)> = <compile_assembly(current_target(), asm, 0, :() -> () @kernel)>
  kgen.return
}

// -----

kgen.generator @func() {
  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // CHECK: constant: string = <"func">
  kgen.param.constant: string = <get_linkage_name(current_target(), :() -> () @func)>
  kgen.return
}

// -----

kgen.generator @no_params() {
  kgen.return
}

kgen.generator @params<a, b>() -> (index, index) {
  %0 = kgen.param.constant = <a>
  %1 = kgen.param.constant = <b>
  kgen.return %0, %1 : index, index
}

kgen.generator @func_param<f: <index, index>() -> (index, index)>() -> index {
  kgen.param.declare bind: () -> (index, index) = <bind_signature(:<index, index>() -> (index, index) f, 7, 9)>
  %0, %1 = kgen.call_param[() -> (index, index): bind]()
  %2 = index.add %0, %1
  kgen.return %2 : index
}

!capture = !kgen.struct<(string, index, (!kgen.pointer<pointer<none>>) capturing -> !kgen.none)>

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() {
  // CHECK-NEXT: constant: struct<(string, index, {{.*}})> = <{ "{{.*}}no_params
  %0 = kgen.param.constant: !capture = <compile_assembly(current_target(), asm, 0, :() -> () @no_params)>
  // CHECK-NEXT: constant: string = <"no_params">
  %1 = kgen.param.constant: string = <get_linkage_name(current_target(), :() -> () @no_params)>
  // CHECK-NEXT: constant: struct<(string, index, {{.*}})> = <{ "{{.*}}params,a=1,b=2
  %2 = kgen.param.constant: !capture = <compile_assembly(current_target(), asm, 0, :() -> (index, index) @params<1, 2>)>
  // CHECK-NEXT: constant: string = <"params,a=1,b=2">
  %3 = kgen.param.constant: string = <get_linkage_name(current_target(), :() -> (index, index) @params<1, 2>)>
  // CHECK-NEXT: constant: struct<(string, index, {{.*}})> = <{ "{{.*}}func_param,f=_params
  %4 = kgen.param.constant: !capture = <compile_assembly(current_target(), asm, 0, :() -> index @func_param<:<index, index>() -> (index, index) @params>)>
  // CHECK-NEXT: constant: string = <"func_param,f=_params">
  %5 = kgen.param.constant: string = <get_linkage_name(current_target(), :() -> index @func_param<:<index, index>() -> (index, index) @params>)>
  kgen.return
}

// -----

!capture = !kgen.struct<(string, index, (!kgen.pointer<none>) capturing -> !kgen.none)>

kgen.generator @lambda() capturing -> index {
  %0 = pop.compiler.global_load "var" : index
  kgen.return %0 : index
}

kgen.generator @captures<f: () capturing -> index>() capturing -> index {
  %0 = kgen.call_param[() capturing -> index: f]()
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() {
  // CHECK-NEXT: struct<(string, index, (!kgen.pointer<none>) capturing -> !kgen.none)> = <{ "{{.*}}", 1, [[POPULATE:@.*]] }>
  %0 = kgen.param.constant: !capture = <compile_assembly(current_target(), asm, 0, :() capturing -> index @captures<:() capturing -> index @lambda>)>
  kgen.return
}

// CHECK: kgen.func [[POPULATE]](%arg0: !kgen.pointer<none>) capturing -> !kgen.none always_inline
// CHECK: [[VAR:%.*]] = pop.compiler.global_load "var" : index
// CHECK: [[ARG:%.*]] = pop.stack_allocation
// CHECK: pop.store [[VAR]], [[ARG]]
// CHECK: [[ARGCAST:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<pointer<none>>
// CHECK: [[PTR:%.*]] = pop.offset [[ARGCAST]][%index0]
// CHECK: [[RAW:%.*]] = pop.pointer.bitcast [[ARG]]
// CHECK: pop.store [[RAW]], [[PTR]]

// -----

kgen.generator @impl(%arg0: !kgen.pointer<i8, apply(:(index) -> index @fwd, 0)>) {
  kgen.return
}

// CHECK-LABEL: kgen.func export @variadic
kgen.generator export @variadic() {
  // CHECK: constant: variadic<(!kgen.pointer<i8>) -> ()> = <[@impl]>
  kgen.param.constant: variadic<(!kgen.pointer<i8, apply(:(index) -> index @fwd, 0)>) -> ()> = <[@impl]>
  kgen.return
}

kgen.generator @fwd(%arg0: index) -> index {
  kgen.return %arg0 : index
}

// -----

// CHECK-LABEL: kgen.func @"decorators,a=1"
// CHCEK-NEXT: decorators<1>

// CHECK-LABEL: kgen.func @"decorators,a=2"
// CHCEK-NEXT: decorators<1>

kgen.generator @decorators<a>()
    decorators<1> {
  kgen.return
}

kgen.generator @elaborate() {
  kgen.call @decorators<1>() : () -> ()
  kgen.call @decorators<2>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @"metadata,x=16,y=16,z=8"
// CHECK-SAME: LLVMMetadata = {nvvm.maxntid = #pop.array<16, 16, 8>
kgen.generator @metadata<x: i32, y: i32, z: i32>() attributes {LLVMMetadata = {
  nvvm.maxntid = #pop.array<x, y, z> : !pop.array<3, i32>
}}{
  kgen.return
}

kgen.generator @kernel() {
  kgen.call @metadata<:i32 16, :i32 16, :i32 8>() : () -> ()
  kgen.return
}

// -----

kgen.generator @func<x>() -> !pop.simd<x, f32> {
  kgen.unreachable
}

kgen.generator @create<T: type>(%arg0: !kgen.paramref<T>) -> !kgen.variant<T, i1> {
  %0 = kgen.variant.create %arg0, 0 : <T, i1>
  kgen.return %0 : !kgen.variant<T, i1>
}

// CHECK-LABEL: kgen.func export @entry
kgen.generator export @entry() {
  // CHECK: constant: variant<<index>() -> !pop.simd<*(0,0), f32>, i1> = <#kgen.variant<:<index>() -> !pop.simd<*(0,0), f32> @func, 0>>
  kgen.param.apply value = [(!kgen.signature<<index>() -> !pop.simd<*(0,0), f32>>) -> !kgen.variant<<index>() -> !pop.simd<*(0,0), f32>, i1>: @create<:type <index>() -> !pop.simd<*(0,0), f32>>](@func)
  kgen.param.constant: variant<<index>() -> !pop.simd<*(0,0), f32>, i1> = <value>
  kgen.return
}

// -----

// CHECK: kgen.func @func
kgen.generator @func() {
  kgen.unreachable
}

// CHECK: kgen.func @"param,a=2"
// CHECK: kgen.func @"param,a=3"
kgen.generator @param<a>() {
  kgen.unreachable
}

// CHECK-LABEL: kgen.func export @entry
kgen.generator export @entry() {
  // CHECK: constant: () -> () = <@func>
  kgen.param.constant: () -> () = <@func>
  // CHECK: constant: () -> () = <@"param,a=2">
  kgen.param.constant: () -> () = <@param<2>>
  // CHECK: constant: struct<(() -> ())> = <{ @"param,a=3" }>
  kgen.param.constant: struct<(() -> ())> = <{ @param<3> }>
  kgen.return
}

// -----

// During elaboration of this example, the type:
//
// <index>(!pop.array<cond(apply(:(index, index) -> i1 @eq, *(0,0), 0), 1, *(0,0)), index>) -> ()
//
// appears in the IR. This type is actually concrete from the perspective of the
// current frame, because it has no parameter expressions. It contains parameter
// operators, but they are part of the signature.
//
// Ensure that this type is valid.

kgen.generator @init<T: type>(%arg0: !kgen.paramref<T>) -> !kgen.struct<()> {
  %struct = kgen.param.constant: struct<()> = <{  }>
  kgen.return %struct : !kgen.struct<()>
}

kgen.generator @eq(%arg0: index, %arg1: index) -> i1 {
  %0 = index.cmp eq(%arg0, %arg1)
  kgen.return %0 : i1
}

kgen.generator @make<x>(%arg0: !pop.array<cond(apply(:(index, index) -> i1 @eq, x, 0), 1, x), index>) {
  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // CHECK-NEXT: constant: struct<()> = <{ }>
  kgen.param.apply lifted = [(!kgen.signature<<index>(!pop.array<cond(apply(:(index, index) -> i1 @eq, *(0,0), 0), 1, *(0,0)), index>) -> ()>) -> !kgen.struct<()>: @init<:type <index>(!pop.array<cond(apply(:(index, index) -> i1 @eq, *(0,0), 0), 1, *(0,0)), index>) -> ()>](@make)
  kgen.param.constant: struct<()> = <lifted>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @"pass_paramref
// CHECK-SAME: () -> !kgen.signature<<index>() -> !pop.simd<apply(:(index) -> index @some_func, *(0,0)), f32>>
kgen.generator @pass_paramref<T: type>() -> !kgen.paramref<T> {
  %0 = kgen.undef : !kgen.paramref<T>
  // CHECK: return %0 : !kgen.signature<<index>() -> !pop.simd<apply(:(index) -> index @some_func, *(0,0)), f32>>
  kgen.return %0 : !kgen.paramref<T>
}

kgen.generator @some_func(%arg0: index) -> index {
  kgen.return %arg0: index
}

kgen.generator @give_func() -> !kgen.signature<(index) -> index>{
  %0 = kgen.param.constant: (index) -> index = <@some_func>
  kgen.return %0 : !kgen.signature<(index) -> index>
}

// CHECK-LABEL: kgen.func @top
kgen.generator @top() {
  kgen.param.apply func = [() -> !kgen.signature<(index) -> index>: @give_func]()
  // CHECK: () -> !kgen.signature<<index>() -> !pop.simd<apply(:(index) -> index @some_func, *(0,0)), f32>>
  kgen.call @pass_paramref<:type <index>() -> !pop.simd<apply(:(index) -> index func, *(0,0)), f32>>() : () -> !kgen.signature<<index>() -> !pop.simd<apply(:(index) -> index func, *(0,0)), f32>>
  kgen.return
}

// -----

// test get_type_method

kgen.generator @indexTraitMethod(%arg0: index) -> index {
  kgen.return %arg0 : index
}

// COM: Check that this gets elaborated to use the concrete function from the vtable below.
// CHECK-LABEL: kgen.func @"generic_call,T=[index{{.*}}]"
kgen.generator @generic_call<T: type>(%arg0: !kgen.paramref<T>) -> index{
  kgen.param.declare traitMethod: (index) -> index  = <get_type_method(T, "traitMethod")>
  %anInt = kgen.param.constant = <1>
  // CHECK: kgen.call @indexTraitMethod
  %result = kgen.call_param[(index) -> index : traitMethod](%anInt)

  kgen.param.declare parametric: <index>() -> ()  = <get_type_method(T, "parametric")>
  // CHECK: kgen.call @"parametricTraitMethod,param=2"
  kgen.call_param[() -> (): bind_signature(:<index>() -> () parametric, 2)]()

  kgen.param.declare bound: () -> () = <get_type_method(T, "bound")>
  // CHECK: kgen.call @"parametricTraitMethod,param=1"
  kgen.call_param[() -> (): bound]()

  kgen.param.declare partial: <index>() -> () = <get_type_method(T, "partial")>
  // CHECK: kgen.call @"twoParameters,parent=11,func=42"
  kgen.call_param[() -> (): bind_signature(:<index>() -> () partial, 11)]()

  kgen.return %result : index
}

kgen.generator @parametricTraitMethod<param>() {
  kgen.return
}

kgen.generator @twoParameters<parent, func>() {
  kgen.return
}

kgen.generator @make_generic_call() -> index {
  %anInt = kgen.param.constant = <1>
  // CHECK: kgen.call @"generic_call,T=[index{{.*}}]"
  %result = kgen.call @generic_call<:type [index, {
    "traitMethod" : (index) -> index = @indexTraitMethod,
    "parametric": <index>() -> () = @parametricTraitMethod,
    "bound": () -> () = @parametricTraitMethod<1>,
    "partial": <index>() -> () = @twoParameters<?, 42>
  }]>(%anInt) : (index) -> index
  kgen.return %result : index
}

// -----

kgen.generator @sizeof<T: type>() -> index {
  %0 = kgen.param.constant: !kgen.int_literal = <get_sizeof(T, current_target())>
  %1 = kgen.int_literal.convert %0 : to index
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func @"self_ref_apply,param=2"
// CHECK-SAME: %arg0: !pop.array<16, i8>
kgen.generator @self_ref_apply<param>(%arg0: !pop.array<apply(:()->index @sizeof<:type array<param, index>>), i8>) {
  kgen.return
}

// CHECK-LABEL: kgen.func export @param_alias
// CHECK-SAME: %arg0: !pop.array<16, i8>
kgen.generator export @param_alias(%arg0: !pop.array<apply(:()->index @sizeof<:type array<2, index>>), i8>) {
  kgen.param.declare fn: <index>(!pop.array<apply(:()->index @sizeof<:type array<*(0,0), index>>), i8>) -> () = <@self_ref_apply>
  // CHECK: call @"self_ref_apply,param=2"(%arg0)
  kgen.call_param[(!pop.array<apply(:()->index @sizeof<:type array<2, index>>), i8>) -> (): bind_signature(:<index>(!pop.array<apply(:()->index @sizeof<:type array<*(0,0), index>>), i8>) -> () fn, 2)](%arg0)
  kgen.return
}

// -----

kgen.generator @f() {
  kgen.return
}

// CHECK-LABEL: kgen.func @"rebind_type,p=1"
kgen.generator @rebind_type<p>(%arg0: !kgen.pointer<array<p, i8>>)
    -> !kgen.pointer<[array<p, i8>, {"f": () -> () = @f}]> {
  // CHECK-NOT: kgen.rebind %arg0
  %0 = kgen.rebind %arg0 : !kgen.pointer<array<p, i8>> to !kgen.pointer<[array<p, i8>, {"f": () -> () = @f}]>
  // CHECK-NEXT: constant: pointer<index> = <store_to_mem(1)>
  kgen.param.constant: pointer<[index, {"f": () -> () = @f}]> = <rebind(:pointer<index> store_to_mem(p))>
  // CHECK-NEXT: constant: pointer<array<1, i8>> = <0>
  kgen.param.constant: pointer<[array<p, i8>, {"f": () -> () = @f}]> = <rebind(:pointer<array<p, i8>> 0)>
  kgen.return %0 : !kgen.pointer<[array<p, i8>, {"f": () -> () = @f}]>
}

kgen.generator @nonparametric_rebind(%arg0: !kgen.pointer<index>) -> index {
  %0 = kgen.rebind %arg0 : !kgen.pointer<index> to !kgen.pointer<[index, {"f": () -> () = @f}]>
  %1 = pop.load %0 : !kgen.pointer<[index, {"f": () -> () = @f}]>
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func @try_rebind
kgen.generator @try_rebind(%arg0: !kgen.pointer<array<1, i8>>) {
  kgen.call @rebind_type<1>(%arg0) : (!kgen.pointer<array<1, i8>>) -> ()
  kgen.param.apply a = [(!kgen.pointer<index>) -> index: @nonparametric_rebind](store_to_mem(1))
  // CHECK: constant = <1>
  kgen.param.constant = <a>
  kgen.return
}

// -----

kgen.generator @fma(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = index.mul %arg1, %arg2
  %1 = index.add %0, %arg0
  kgen.return %1 : index
}

!capture = !kgen.struct<(string, index, (!kgen.pointer<pointer<none>>) capturing -> !kgen.none)>

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() {
  // CHECK: mul i64
  // CHECK: add i64
  %0 = kgen.param.constant: !capture = <compile_assembly(current_target(), llvm, 0, :(index, index, index) -> (index) @fma)>
  kgen.return
}

// -----

kgen.generator @might_fail<succeed: i1>() {
  kgen.param.assert <succeed>, "did not succeed"
  kgen.return
}

!capture = !kgen.struct<(string, index, (!kgen.pointer<pointer<none>>) capturing -> !kgen.none)>

// CHECK-LABEL: @compile_assembly_conditional
kgen.generator export @compile_assembly_conditional() {
  // CHECK-NEXT: <{ "failed {{.*}}", -1, *? }>
  kgen.param.constant: !capture = <compile_assembly(current_target(), llvm, 1, :() -> () @might_fail<:i1 0>)>
  // CHECK-NEXT: <{ "{{.*}}", 0, [[CAP_FN:@.*]] }>
  kgen.param.constant: !capture = <compile_assembly(current_target(), llvm, 1, :() -> () @might_fail<:i1 1>)>
  kgen.return
}

// CHECK: kgen.func [[CAP_FN]]

// -----

// CHECK-NOT: @no_impl
kgen.generator @no_impl() -> index {
  kgen.param.assert <0>, "bad"
  %index0 = kgen.param.constant = <0>
  kgen.return %index0 : index
}

kgen.generator @make_true() -> i1 {
  %0 = kgen.param.constant: i1 = <1>
  kgen.return %0 : i1
}

kgen.generator export @conditional_alias() {
  kgen.param.declare value = <cond(apply(:() -> i1 @make_true), 1, apply(:() -> index @no_impl))>
  kgen.return
}

// -----

kgen.generator @call_it(%arg0: !kgen.signature<() -> index>) -> index {
  %0 = kgen.call_indirect %arg0() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @"give,a=1"
kgen.generator @give<a>() -> index {
  %idx0 = kgen.param.constant = <a>
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func export @apply_expr
kgen.generator export @apply_expr() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:(!kgen.signature<() -> index>) -> index @call_it, @give<1>)>
  kgen.return
}

// CHECK-LABEL: kgen.func export @apply_op
kgen.generator export @apply_op() {
  // CHECK-NEXT: <2>
  kgen.param.apply x = [(!kgen.signature<() -> index>) -> index: @call_it](@give<2>)
  kgen.param.constant = <x>
  kgen.return
}

// -----

kgen.generator @fwd_type(%arg0: !kgen.type) -> !kgen.type {
  kgen.return %arg0 : !kgen.type
}

kgen.generator @fwd_sig(%arg0: !kgen.signature<<type>(!kgen.paramref<apply(:(!kgen.type)->!kgen.type @fwd_type, *(0,0))>) -> ()>) -> !kgen.signature<<type>(!kgen.paramref<apply(:(!kgen.type)->!kgen.type @fwd_type, *(0,0))>) -> ()> {
  kgen.return %arg0 :  !kgen.signature<<type>(!kgen.paramref<apply(:(!kgen.type)->!kgen.type @fwd_type, *(0,0))>) -> ()>
}

kgen.generator @fn<T: type>(%arg0: !kgen.paramref<apply(:(!kgen.type) -> !kgen.type @fwd_type, T)>) {
  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // COM: Just check that the parameter expression can be resolved.
  kgen.param.declare f: <type>(!kgen.paramref<apply(:(!kgen.type)->!kgen.type @fwd_type, *(0,0))>) -> () = <apply(:(!kgen.signature<<type>(!kgen.paramref<apply(:(!kgen.type)->!kgen.type @fwd_type, *(0,0))>) -> ()>) -> !kgen.signature<<type>(!kgen.paramref<apply(:(!kgen.type)->!kgen.type @fwd_type, *(0,0))>) -> ()> @fwd_sig, @fn)>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @conflicting_values
kgen.generator @conflicting_values() {
  // CHECK-NEXT: constant = <1>
  // CHECK-NEXT: constant = <2>
  kgen.param.declare b: i1 = <1>
  kgen.param.if <b> {
    kgen.param.declare a = <1>
    kgen.param.constant = <a>
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  kgen.param.if <b> {
    kgen.param.declare a = <2>
    kgen.param.constant = <a>
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func export @param_for
kgen.generator export @param_for(%arg0: i1, %arg1: index) {
  kgen.call @sum_from_zero<0>() : () -> ()
  kgen.call @sum_from_zero<1>() : () -> ()
  kgen.call @sum_from_zero<2>() : () -> ()

  kgen.call @terminators(%arg0) : (i1) -> ()
  kgen.call @nested_param_for() : () -> ()
  kgen.call @parametric_result() : () -> ()

  kgen.call @for_else<0>(%arg1) : (index) -> index
  kgen.call @for_else<1>(%arg1) : (index) -> index
  kgen.return
}

// CHECK-LABEL: kgen.func @"sum_from_zero,upper=0"
// CHECK-NEXT:    %idx0 = index.constant 0
// CHECK-NEXT:    %0 = hlcf.loop "param_for_outer" () -> index
// CHECK-NEXT:      hlcf.break "param_for_outer" %idx0
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0

// CHECK-LABEL: kgen.func @"sum_from_zero,upper=1"
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %0 = hlcf.loop "param_for_outer" () -> index
// CHECK-NEXT:     %1 = hlcf.loop () -> index
// CHECK-NEXT:       %index1 = kgen.param.constant = <1>
// CHECK-NEXT:       %2 = index.add %idx0, %index1
// CHECK-NEXT:       break %2 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     break "param_for_outer" %1 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : index

// CHECK-LABEL: kgen.func @"sum_from_zero,upper=2"
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %0 = hlcf.loop "param_for_outer" () -> index
// CHECK-NEXT:     %1 = hlcf.loop () -> index
// CHECK-NEXT:       %index2 = kgen.param.constant = <2>
// CHECK-NEXT:       %3 = index.add %idx0, %index2
// CHECK-NEXT:       hlcf.break %3 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = hlcf.loop () -> index
// CHECK-NEXT:       %index1 = kgen.param.constant = <1>
// CHECK-NEXT:       %3 = index.add %1, %index1
// CHECK-NEXT:       hlcf.break %3 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     hlcf.break "param_for_outer" %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   kgen.return %0 : index

kgen.generator @sum_from_zero<upper>() -> index {
  %idx0 = index.constant 0
  %0 = kgen.param.for i in upper iter :(!kgen.pointer<index>, !kgen.pointer<struct<(index, index, i1)>> byref_result) -> !kgen.none @count_to_zero
    (%arg0 = %idx0 : index) -> index {
    %1 = kgen.param.constant = <i>
    %2 = index.add %arg0, %1
    kgen.param.for.continue %2 : index
  } else (%arg1: index) {
    kgen.param.yield %arg1 : index
  }
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @terminators
kgen.generator @terminators(%arg0: i1) {
  // CHECK-NEXT: hlcf.loop "param_for_outer" {
  // CHECK-NEXT:   hlcf.loop {
  // CHECK-NEXT:     hlcf.if %arg0 {
  // CHECK-NEXT:       kgen.return
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       hlcf.break
  // CHECK-NEXT:     }
  // CHECK-NEXT:     hlcf.break "param_for_outer"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   hlcf.break "param_for_outer"
  // CHECK-NEXT: }
  kgen.param.for i in 1 iter :(!kgen.pointer<index>, !kgen.pointer<struct<(index, index, i1)>> byref_result) -> !kgen.none @count_to_zero {
    hlcf.if %arg0 {
      kgen.return
    } else {
      kgen.param.for.continue
    }
    kgen.param.for.break
  } else {
    kgen.param.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @nested_param_for
kgen.generator @nested_param_for() {
  // CHECK: { 2, 2 }
  // CHECK: { 2, 1 }
  // CHECK: { 1, 2 }
  // CHECK: { 1, 1 }
  kgen.param.for i in 2 iter :(!kgen.pointer<index>, !kgen.pointer<struct<(index, index, i1)>> byref_result) -> !kgen.none @count_to_zero {
    kgen.param.for j in 2 iter :(!kgen.pointer<index>, !kgen.pointer<struct<(index, index, i1)>> byref_result) -> !kgen.none @count_to_zero {
      kgen.param.constant: struct<(index, index)> = <{ i, j }>
      kgen.param.for.continue
    } else {
      kgen.param.yield
    }
    kgen.param.if <0> {
      kgen.param.yield
    } else {
      kgen.param.yield
    }
    kgen.param.for.continue
  } else {
    kgen.param.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @"for_else,upper=0"
// CHECK-NEXT:    %0 = hlcf.loop "param_for_outer" () -> index
// CHECK-NEXT:      %index1 = kgen.param.constant = <1>
// CHECK-NEXT:      %1 = index.add %index1, %arg0
// CHECK-NEXT:      hlcf.break "param_for_outer" %1
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0

// CHECK-LABEL: kgen.func @"for_else,upper=1"
// CHECK-NEXT:    %0 = hlcf.loop "param_for_outer" () -> index
// CHECK-NEXT:      %1 = hlcf.loop () -> index
// CHECK-NEXT:        break %arg0
// CHECK-NEXT:      }
// CHECK-NEXT:      %index2 = kgen.param.constant = <2>
// CHECK-NEXT:      %2 = index.add %index2, %1
// CHECK-NEXT:      break "param_for_outer" %2
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0

kgen.generator @for_else<upper>(%arg0: index) -> index {
  kgen.param.declare param = <add(upper, 1)>
  %0 = kgen.param.for i in upper iter :(!kgen.pointer<index>, !kgen.pointer<struct<(index, index, i1)>> byref_result) -> !kgen.none @count_to_zero
    (%arg1 = %arg0 : index) -> index {
    kgen.param.for.continue %arg1 : index
  } else (%arg1: index) {
    %1 = kgen.param.constant = <param>
    %2 = index.add %1, %arg1
    kgen.param.yield %2 : index
  }
  kgen.return %0 : index
}

// COM: Just check that this passes the verifier.
kgen.generator @parametric_result() {
  kgen.param.declare a = <1>
  %0 = kgen.param.constant: array<a, i1> = <?>
  %1 = kgen.param.for i in 1 iter :(!kgen.pointer<index>, !kgen.pointer<struct<(index, index, i1)>> byref_result) -> !kgen.none @count_to_zero
    (%arg0 = %0 : !pop.array<a, i1>) -> !pop.array<a, i1> {
    kgen.param.for.continue %arg0 : !pop.array<a, i1>
  } else (%arg0: !pop.array<a, i1>) {
    kgen.param.yield %arg0 : !pop.array<a, i1>
  }
  kgen.return
}

kgen.generator @count_to_zero(%arg0: !kgen.pointer<index>, %arg1: !kgen.pointer<struct<(index, index, i1)>> byref_result) -> !kgen.none {
  %i0 = pop.load %arg0 : !kgen.pointer<index>
  %none = kgen.param.constant: none = <#kgen.none>
  %idx0 = index.constant 0
  %0 = index.cmp eq(%idx0, %i0)
  hlcf.if %0 {
    %struct = kgen.param.constant: struct<(index, index, i1)> = <{ 0, 0, 1 }>
    pop.store %struct, %arg1 : !kgen.pointer<struct<(index, index, i1)>>
    kgen.return %none : !kgen.none
  } else {
    hlcf.yield
  }
  %idx1 = index.constant 1
  %1 = index.sub %i0, %idx1
  %false = index.bool.constant false
  %2 = kgen.struct.create(%1, %i0, %false) : !kgen.struct<(index, index, i1)>
  pop.store %2, %arg1 : !kgen.pointer<struct<(index, index, i1)>>
  kgen.return %none : !kgen.none
}

// -----

kgen.generator @forward_pointer(%arg0: !kgen.pointer<index>) -> !kgen.pointer<index> {
  kgen.return %arg0 : !kgen.pointer<index>
}

// CHECK-LABEL: kgen.func export @load_from_mem
kgen.generator export @load_from_mem() {
  // CHECK-NEXT: <42>
  kgen.param.constant = <load_from_mem(:pointer<index> apply(:(!kgen.pointer<index>) -> !kgen.pointer<index> @forward_pointer, store_to_mem(42)))>
  kgen.return
}


// -----

// MOCO-942 - Elaborator making wild symbol names

// This should not get the full vtable substituted in.
// CHECK: kgen.func @"genA,type_param=::string_literal::StringLiteral"() no_inline {
// CHECK: kgen.func @"genA,type_param=x::y::z"() no_inline { 
kgen.generator @genA<type_param: type>() no_inline {
  kgen.return
}

kgen.generator @"::string_literal::StringLiteral::__del__(::string_literal::StringLiteral)_thunk"(%arg0: !kgen.pointer<string> owned_in_mem) -> !kgen.none always_inline_no_debug {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @"x::y::z::__del__(x::y::z)"(%arg0: !kgen.pointer<string> owned_in_mem) -> !kgen.none always_inline_no_debug {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @call_generator_test() {
  kgen.call @genA<:type [string, {"__del__" : (!kgen.pointer<string> owned_in_mem) -> !kgen.none = @"::string_literal::StringLiteral::__del__(::string_literal::StringLiteral)_thunk"}]>() : () -> ()
  kgen.call @genA<:type [string, {"__del__" : (!kgen.pointer<string> owned_in_mem) -> !kgen.none = @"x::y::z::__del__(x::y::z)"}]>() : () -> ()
  kgen.return
}

