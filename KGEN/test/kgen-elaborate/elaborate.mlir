// RUN: kgen-opt %s -elaborate-generators="search-path=%S" -allow-unregistered-dialect | FileCheck %s

kgen.include "library-test.mlir"

// This is left untouched.
// CHECK-LABEL: kgen.func @test0<() -> index>() -> index {
// CHECK-NEXT: %[[V0:.*]] = kgen.param.constant = <1>
// CHECK-NEXT:  kgen.return<123456> %[[V0]] : index
// CHECK-NEXT: }
kgen.func @test0<() -> index>() -> index {
  %0 = kgen.param.constant = <1>
  kgen.return <123456> %0 : index
}

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
  // CHECK-NEXT: %{{.*}} = kgen.param.constant : dtype = <f32>
  %3 = kgen.param.constant : !kgen.dtype = <type1>

  // CHECK-NEXT: kgen.return
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

kgen.generator @genA<size, type: dtype, val: f32 -> index>(%arg0: si32) -> si32 {

  %0 = kgen.param.constant = <add(size, 4)>
  %1 = kgen.param.constant : dtype = <type>
  %2 = kgen.param.constant : f32 = <val>

  // Silly op so we know when something used this.
  "genA op"() { value = #kgen.param.decl.ref<size> : index} : () -> !meta.scalar<type>

  kgen.return<mul(size, 2)> %arg0 : si32
}
// CHECK-LABEL: kgen.func @"genA,size=42,type=f32,val=2"<() -> index>
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:   %[[V0:.*]] = kgen.param.constant  = <46>
// CHECK-NEXT:   %[[V1:.*]] = kgen.param.constant : dtype = <f32>
// CHECK-NEXT:   %[[V2:.*]] = kgen.param.constant : f32 = <2.000000e+00>
// CHECK-NEXT:   %[[V3:.*]] = "genA op"() {value = 42 : index} : () -> !meta.scalar<f32>
// CHECK-NEXT:   kgen.return<84> %[[ARG0]] : si32
// CHECK-NEXT: }

// CHECK-LABEL: kgen.func @"genA,size=19,type=si8,val=1.5"<() -> index>
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:    %[[V0:.*]] = kgen.param.constant  = <23>
// CHECK-NEXT:    %[[V1:.*]] = kgen.param.constant : dtype = <si8>
// CHECK-NEXT:    %[[V2:.*]] = kgen.param.constant : f32 = <1.500000e+00>
// CHECK-NEXT:    %[[V3:.*]] = "genA op"() {value = 19 : index} : () -> !meta.scalar<si8>
// CHECK-NEXT:    kgen.return<38> %[[ARG0]] : si32
// CHECK-NEXT:  }

// CHECK-LABEL: kgen.func @call_generator_test
// CHECK-SAME: %[[ARG0:.*]]: si32, %[[ARG1:.*]]: si32
kgen.generator @call_generator_test(%arg0: si32, %arg1: si32)
   -> (si32, si32, si32, index, index) {
  // Can invoke the generator directly.
  %0 = kgen.call @trivial_generator(%arg0) : (si32) -> si32
  // CHECK-NEXT: %{{.*}} = kgen.call @trivial_generator(%[[ARG0]])

  // CHECK-NOT: kgen.param.declare
  kgen.param.declare our_size = <42>

  // Can invoke parameterized generators directly.
  %1 = kgen.call @genA<size = our_size, type : dtype = f32, val : f32 = 2.0 -> resultSizeA>(%arg0) : (si32) -> si32
  // CHECK-NEXT: %{{.*}} = kgen.call @"genA,size=42,type=f32,val=2"<() -> resultSizeA>(%[[ARG0]]) : (si32) -> si32

  %2 = kgen.call @genA<size = 19, type : dtype = si8, val : f32 = 1.5 -> resultSizeB>(%arg1) : (si32) -> si32
  // CHECK-NEXT: %{{.*}} = kgen.call @"genA,size=19,type=si8,val=1.5"<() -> resultSizeB>(%[[ARG1]]) : (si32) -> si32

  %3 = kgen.call @genA<size = 19, type : dtype = si8, val : f32 = 1.5 -> resultSizeC>(%arg1) : (si32) -> si32
  // CHECK-NEXT: %{{.*}} = kgen.call @"genA,size=19,type=si8,val=1.5"<() -> resultSizeC>(%[[ARG1]]) : (si32) -> si32


  %4 = kgen.param.constant = <resultSizeA>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <84>

  %5 = kgen.param.constant = <resultSizeB>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <38>

  %6 = kgen.param.constant = <resultSizeC>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <38>

  %7 = kgen.call @test0<() -> kernelResult>() : () -> index
  // CHECK-NEXT: %{{.*}} = kgen.call @test0<() -> kernelResult>()

  %8 = kgen.param.constant = <kernelResult>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <123456>

  kgen.return %0, %1, %2, %4, %5 : si32, si32, si32, index, index
}

//===----------------------------------------------------------------------===//

// CHECK-NOT: kgen.generator.interface @genItf
kgen.generator.interface @genItf<x -> index>(si32) -> si32

// CHECK-LABEL: kgen.func @"genItf_impl1,x=42"<() -> index>(
// CHECK-SAME: %[[ARG0:.*]]: si32
// CHECK-NEXT:   "genItf_impl1"() {value = 42 : index} : () -> ()
// CHECK-NEXT:   kgen.return<43> %[[ARG0]] : si32
// CHECK-NEXT: }
kgen.generator @genItf_impl1<x -> index>(%arg0: si32) -> si32
  implements @genItf {
  "genItf_impl1"() { value = #kgen.param.decl.ref<x> : index} : () -> ()
  kgen.return<add(x, 1)> %arg0 : si32
}

// CHECK-LABEL: kgen.func @"genItf_impl2,x=42"<() -> index>(
// CHECK-SAME: %[[ARG0:.*]]: si32
// CHECK-NEXT:   "genItf_impl2"() {value = 42 : index} : () -> ()
// CHECK-NEXT:   kgen.return<84> %[[ARG0]] : si32
// CHECK-NEXT: }
kgen.generator @genItf_impl2<x -> index>(%arg0: si32) -> si32
  implements @genItf {
  "genItf_impl2"() { value = #kgen.param.decl.ref<*"x"> : index} : () -> ()
  kgen.return<mul(x, 2)> %arg0 : si32
}

// CHECK-LABEL: kgen.func @use_interface(
// CHECK-SAME: %[[ARG0:.*]]: si32
// CHECK-NEXT: %[[V0:.*]] = kgen.call @"genItf_impl1,x=42"<() -> out>(%[[ARG0]])
// CHECK-NEXT: %[[V1:.*]] = kgen.param.constant = <43>

// CHECK-LABEL: kgen.func @use_interface_concrete_0
// CHECK-SAME: %[[ARG0:.*]]: si32
// CHECK-NEXT:    %{{.*}} = kgen.call @"genItf_impl2,x=42"<() -> out>(%[[ARG0]]) : (si32) -> si32
// CHECK-NEXT:     %{{.*}} = kgen.param.constant = <84>
kgen.generator @use_interface(%arg0: si32) -> index {
  %0 = kgen.call @genItf<x = 42 -> out>(%arg0) : (si32) -> si32
  %1 = kgen.param.constant = <out>
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func @use_using_interface
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> index {
// CHECK-NEXT:   %[[V0:.*]] = kgen.call @use_interface(%[[ARG0]]) : (si32) -> index
// CHECK-NEXT:   kgen.return %[[V0]] : index

// CHECK-LABEL: kgen.func @use_using_interface_concrete_1
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> index {
// CHECK-NEXT:   %[[V0:.*]] = kgen.call @use_interface_concrete_0(%[[ARG0]]) : (si32) -> index
// CHECK-NEXT:   kgen.return %[[V0]] : index
kgen.generator @use_using_interface(%arg0: si32) -> index {
  %0 = kgen.call @use_interface(%arg0) : (si32) -> index
  kgen.return %0 : index
}

//===----------------------------------------------------------------------===//

// CHECK-NOT: @genItf2<x>()
kgen.generator.interface @genItf2<x>()

// CHECK-NOT: kgen.func @"genItf2_impl0,x=1"() {
// CHECK-LABEL: kgen.func @"genItf2_impl0,x=0"() {
// CHECK-NEXT:   "impl0"() : () -> ()
// CHECK-NEXT:   kgen.return
// CHECK-NOT: kgen.func @"genItf2_impl0,x=1"() {
kgen.generator @genItf2_impl0<x>()
  constraints <[eq(x, 0), "x must be zero"]> implements @genItf2 {
  "impl0"() : () -> ()
  kgen.return
}

// CHECK-NOT: kgen.func @"genItf2_impl1,x=0"()
// CHECK-LABEL: kgen.func @"genItf2_impl1,x=1"() {
// CHECK-NEXT:   "impl1"() : () -> ()
// CHECK-NEXT:   kgen.return
// CHECK-NOT: kgen.func @"genItf2_impl1,x=0"()
kgen.generator @genItf2_impl1<x>()
  constraints <[eq(x, 1), "x must be 1"]> implements @genItf2 {
  "impl1"() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @use_Itf2zero() {
// CHECK-NEXT:   kgen.call @"genItf2_impl0,x=0"() : () -> ()
// CHECK-NEXT:   kgen.return
kgen.generator @use_Itf2zero() {
  kgen.call @genItf2<x = 0>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @use_Itf2one() {
// CHECK-NEXT:   kgen.call @"genItf2_impl1,x=1"() : () -> ()
// CHECK-NEXT:   kgen.return
// CHECK-NEXT: }
kgen.generator @use_Itf2one() {
  kgen.call @genItf2<x = 1>() : () -> ()
  kgen.return
}

// -----

kgen.generator.interface @genItf3<ty: dtype>()

// This implementation is fine.
// CHECK-LABEL: kgen.func @"genItf3_impl0,ty=f32"() {
kgen.generator @genItf3_impl0<ty: dtype>() implements @genItf3 {
  "impl0"() : () -> ()
  kgen.return
}

// This generates a kernel that fails to verify, so it isn't used and must be
// deleted.
// CHECK-NOT: genItf3_impl1
kgen.generator @genItf3_impl1<ty: dtype>() implements @genItf3 {
  %0 = pop.constant(1.0 : f32) : !meta.scalar<ty>
  %1 = meta.cast_to_builtin %0: !meta.scalar<ty> to i8
  kgen.return
}

// This has a single viable implementation.
// CHECK-LABEL: kgen.func @use_Itf3() {
// CHECK-NEXT:    kgen.call @"genItf3_impl0,ty=f32"()
kgen.generator @use_Itf3() {
  kgen.call @genItf3<ty: dtype = f32>() : () -> ()
  kgen.return
}

// -----

// Test that expansions are tracked and each ultimate kernel version only allows
// any particular generator/parameter set pair to expand one direction, reducing
// exponential explosion.

// CHECK-LABEL: kgen.func @track_expansions
// CHHECK-SAME: (%[[ARG0:.*]]: si32)
// CHECK-NEXT: kgen.call @"genItf_impl1,x=42"<() -> out>(%[[ARG0]]) : (si32) -> si32
// CHECK-NEXT: kgen.call @"genItf_impl1,x=42"<() -> out1>(%[[ARG0]]) : (si32) -> si32
// CHECK-NEXT: kgen.call @use_interface(%[[ARG0]])

// CHECK-NOT: kgen.func @track_expansions

// CHECK-LABEL: kgen.func @track_expansions_concrete_2
// CHECK-SAME: (%[[ARG0:.*]]: si32)
// CHECK-NEXT: kgen.call @"genItf_impl2,x=42"
// CHECK-NEXT: kgen.call @"genItf_impl2,x=42"
// CHECK-NEXT: kgen.call @use_interface_concrete_0(%[[ARG0]])

// CHECK-NOT: kgen.func @track_expansions

kgen.generator @track_expansions(%arg0: si32) {
  // Within any generated kernel genItf should expand the same way.
  %0 = kgen.call @genItf<x = 42 -> out>(%arg0) : (si32) -> si32
  %1 = kgen.call @genItf<x = 42 -> out1>(%arg0) : (si32) -> si32

  // Even if deeply nested within other generator/kernel invocations
  %2 = kgen.call @use_interface(%arg0) : (si32) -> index
  kgen.return
}


// -----

// Test that parameter and result argument types get rewritten and specialized.

// CHECK-LABEL: kgen.func @"float_constant_f32,value=1.5,type=f32"() -> !meta.scalar<f32> {
// ...
// CHECK:    %[[V1:.*]] = llvm.fptrunc
// CHECK:    %[[V2:.*]] = meta.cast_from_builtin %[[V1]] : f32 to !meta.scalar<f32>
// CHECK:    kgen.return %[[V2]] : !meta.scalar<f32>

kgen.generator @float_constant_f32<value: f64, type: dtype>() -> !meta.scalar<type>
  constraints <[eq(:dtype type, f32), "float please"]>  {
  %0 = kgen.param.constant : f64 = <value>
  %1 = llvm.fptrunc %0 : f64 to f32
  %2 = meta.cast_from_builtin %1: f32 to !meta.scalar<type>
  kgen.return %2 : !meta.scalar<type>
}

// CHECK-LABEL: kgen.func @test_f32() -> f32 {
// CHECK:    %[[V0:.*]] = kgen.call @"float_constant_f32,value=1.5,type=f32"() : () -> !meta.scalar<f32>
// CHECK:    %[[V1:.*]] = meta.cast_to_builtin %[[V0]] : !meta.scalar<f32> to f32
kgen.generator @test_f32() -> f32 {
  kgen.param.declare type : dtype = <f32>
  %1 = kgen.call @float_constant_f32<value: f64 = 1.5, type: dtype = type>() : () -> !meta.scalar<type>
  %2 = meta.cast_to_builtin %1 : !meta.scalar<type> to f32
  kgen.return %2 : f32
}

// -----

// Test that we can do static assertions on computed parameter expressions (e.g.
// those that are the result of a sub-generator invocation.

kgen.generator.interface @getSIMDLength<dt: dtype -> index>()

kgen.generator @getSIMDLengthF32<dt: dtype -> index>()
     implements @getSIMDLength {
  // This could be implemented as a constraint.
  kgen.param.assert <eq(:dtype dt, f32)>, "this only works for f32"
  // vector length for floats is 4 on our target.
  kgen.return <4>
}

kgen.generator @getSIMDLengthF64<dt: dtype -> index>()
     implements @getSIMDLength {
  // This could be implemented as a constraint.
  kgen.param.assert <eq(:dtype dt, f64)>, "this only works for f32"
  // vector length for doubles is 2 on our target.
  kgen.return <2>
}

// CHECK-LABEL: kgen.func @paramAssertExample()
// CHECK-NEXT:    kgen.call @"getSIMDLengthF32,dt=f32"<() -> flen>()
// CHECK-NEXT:    kgen.return
kgen.generator @paramAssertExample() {
  kgen.call @getSIMDLength<dt : dtype = f32 -> flen>() : () -> ()

  // Should succeed.
  kgen.param.assert <eq(flen, 4)>, "vector length should be 4 for floats"
  kgen.return
}

// CHECK-LABEL: kgen.func @parametricTypes(
kgen.generator @parametricTypes(%arg0: !meta.scalar<ui64>, %arg1: !meta.simd<2, f32>) {
  kgen.param.declare dt: dtype = <ui32>
  kgen.param.declare ty1: type = <!meta.scalar<dt>>

  // CHECK-NEXT:   "impl0"() : () -> !meta.scalar<ui32>
  "impl0"() : () -> !kgen.paramref<ty1>

  // CHECK-NEXT: = kgen.call @"parametricAdd,ty=!meta.scalar<ui64>"
  // CHECK-SAME: (%[[ARG0:.*]], %[[ARG0:.*]]) : (!meta.scalar<ui64>, !meta.scalar<ui64>) -> !meta.scalar<ui64>
  %0 = kgen.call @parametricAdd<ty: type = !meta.scalar<ui64>>(%arg0, %arg0) : (!meta.scalar<ui64>, !meta.scalar<ui64>) -> !meta.scalar<ui64>

  // CHECK-NEXT: = kgen.call @"parametricAdd,ty=!meta.simd<2, f32>"(%[[ARG1]], %[[ARG1]]) : (!meta.simd<2, f32>, !meta.simd<2, f32>) -> !meta.simd<2, f32>
  %1 = kgen.call @parametricAdd<ty: type = !meta.simd<2, f32>>(%arg1, %arg1) : (!meta.simd<2, f32>, !meta.simd<2, f32>) -> !meta.simd<2, f32>

  kgen.return
}

// CHECK-LABEL: kgen.func @"parametricAdd,ty=!meta.scalar<ui64>"
// CHECK-SAME: (%[[ARG0:.*]]: !meta.scalar<ui64>, %[[ARG1:.*]]: !meta.scalar<ui64>) -> !meta.scalar<ui64> {
// CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !meta.scalar<ui64>
// CHECK-NEXT: kgen.return %[[V0]] : !meta.scalar<ui64>

// CHECK-LABEL: kgen.func @"parametricAdd,ty=!meta.simd<2, f32>"
// CHECK-SAME: (%[[ARG0:.*]]: !meta.simd<2, f32>, %[[ARG1:.*]]: !meta.simd<2, f32>) -> !meta.simd<2, f32> {
// CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !meta.simd<2, f32>
// CHECK-NEXT: kgen.return %[[V0]] : !meta.simd<2, f32>

kgen.generator @parametricAdd<ty: type>
  (%a: !kgen.paramref<ty>, %b: !kgen.paramref<ty>) -> !kgen.paramref<ty> {
  %res = pop.add %a, %b : !kgen.paramref<ty>
  kgen.return %res : !kgen.paramref<ty>
}

// CHECK-LABEL: kgen.func @"takeUnary,dt=si32,fn=@doubleExample"()
// CHECK: %0 = kgen.call @"doubleExample,dt=si32"(%cst) : (!meta.scalar<si32>) -> !meta.scalar<si32>
// CHECK:  %1 = kgen.call @"doubleExample,dt=si32"(%0) : (!meta.scalar<si32>) -> !meta.scalar<si32>

// CHECK-LABEL: kgen.func @"takeUnary,dt=f32,fn=@nopExample"() {
// CHECK:    %0 = kgen.call @"nopExample,dt=f32"(%cst) : (!meta.scalar<f32>) -> !meta.scalar<f32>
// CHECK:    %1 = kgen.call @"nopExample,dt=f32"(%0) : (!meta.scalar<f32>) -> !meta.scalar<f32>

kgen.generator @takeUnary
  <dt: dtype, fn: signature<<dt:dtype>(!meta.scalar<dt>) -> !meta.scalar<dt>>>() {

  %0 = pop.constant(1) : !meta.scalar<dt>
  %1 = kgen.call_param[signature<<dt:dtype>(!meta.scalar<dt>) -> !meta.scalar<dt>>: fn]<dt: dtype = dt>(%0) 
  %2 = kgen.call_param[signature<<dt:dtype>(!meta.scalar<dt>) -> !meta.scalar<dt>>: fn]<dt: dtype = dt>(%1)
  kgen.return
}

kgen.generator @doubleExample<dt:dtype>(%arg0: !meta.scalar<dt>) -> !meta.scalar<dt> {
  %0 = pop.add %arg0, %arg0: !meta.scalar<dt>
  kgen.return %0 : !meta.scalar<dt>
}

kgen.generator @nopExample<dt:dtype>(%arg0: !meta.scalar<dt>) -> !meta.scalar<dt> {
  kgen.return %arg0 : !meta.scalar<dt>
}

kgen.generator @takeParametricBinary
  <dt: dtype,
   fn: signature<<ty: type>(!kgen.paramref<ty>, !kgen.paramref<ty>) -> !kgen.paramref<ty>>
  >() {

  %0 = pop.constant(1) : !meta.scalar<dt>
  // TODO: call_param type checking not correct.
  //%1 = kgen.call_param[signature<<ty: type>(!kgen.paramref<ty>, !kgen.paramref<ty>) -> !kgen.paramref<ty>>: fn]<ty: type = !meta.scalar<dt>>(%0, %0) 
  kgen.return
}

// CHECK-LABEL:  kgen.func @test_region() {
kgen.generator @test_region() {
  // CHECK: kgen.call @"takeUnary,dt=si32,fn=@doubleExample"()
  kgen.call @takeUnary<dt: dtype = si32,
     fn : signature<<dt:dtype>(!meta.scalar<dt>) -> !meta.scalar<dt>> = @doubleExample>() : () -> ()

  // CHECK: kgen.call @"takeUnary,dt=f32,fn=@nopExample"()
  kgen.call @takeUnary<dt: dtype = f32,
     fn : signature<<dt:dtype>(!meta.scalar<dt>) -> !meta.scalar<dt>> = @nopExample>() : () -> ()

  // CHECK: kgen.call @"takeParametricBinary,dt=f32,fn=@parametricAdd"()
  kgen.call @takeParametricBinary<dt: dtype = f32,
      fn : signature<<ty: type>(!kgen.paramref<ty>, !kgen.paramref<ty>) -> !kgen.paramref<ty>>
      = @parametricAdd>() : () -> ()

  kgen.return 
}


