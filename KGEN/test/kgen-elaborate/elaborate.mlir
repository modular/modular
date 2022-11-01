// RUN: kgen-opt %s -split-input-file -elaborate-generators="search-path=%S" -allow-unregistered-dialect | FileCheck %s

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
  // CHECK-NEXT: %{{.*}} = kgen.param.constant: dtype = <f32>
  %3 = kgen.param.constant: !kgen.dtype = <type1>

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
  %1 = kgen.param.constant: dtype = <type>
  %2 = kgen.param.constant: f32 = <val>

  // Silly op so we know when something used this.
  "genA op"() { value = #kgen.param.decl.ref<"size"> : index} : () -> !pop.scalar<type>

  kgen.return<mul(size, 2)> %arg0 : si32
}
// CHECK-LABEL: kgen.func @"genA,size=42,type=f32,val=2"<() -> index>
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:   %[[V0:.*]] = kgen.param.constant  = <46>
// CHECK-NEXT:   %[[V1:.*]] = kgen.param.constant: dtype = <f32>
// CHECK-NEXT:   %[[V2:.*]] = kgen.param.constant: f32 = <2.000000e+00>
// CHECK-NEXT:   %[[V3:.*]] = "genA op"() {value = 42 : index} : () -> !pop.scalar<f32>
// CHECK-NEXT:   kgen.return<84> %[[ARG0]] : si32
// CHECK-NEXT: }

// CHECK-LABEL: kgen.func @"genA,size=19,type=si8,val=1.5"<() -> index>
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:    %[[V0:.*]] = kgen.param.constant  = <23>
// CHECK-NEXT:    %[[V1:.*]] = kgen.param.constant: dtype = <si8>
// CHECK-NEXT:    %[[V2:.*]] = kgen.param.constant: f32 = <1.500000e+00>
// CHECK-NEXT:    %[[V3:.*]] = "genA op"() {value = 19 : index} : () -> !pop.scalar<si8>
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
  "genItf_impl1"() { value = #kgen.param.decl.ref<"x"> : index} : () -> ()
  kgen.return<add(x, 1)> %arg0 : si32
}

// CHECK-LABEL: kgen.func @"genItf_impl2,x=42"<() -> index>(
// CHECK-SAME: %[[ARG0:.*]]: si32
// CHECK-NEXT:   "genItf_impl2"() {value = 42 : index} : () -> ()
// CHECK-NEXT:   kgen.return<84> %[[ARG0]] : si32
// CHECK-NEXT: }
kgen.generator @genItf_impl2<x -> index>(%arg0: si32) -> si32
  implements @genItf {
  "genItf_impl2"() { value = #kgen.param.decl.ref<"x"> : index} : () -> ()
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

//===----------------------------------------------------------------------===//

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
  %0 = pop.constant(1.0 : f32) : !pop.scalar<ty>
  %1 = pop.cast_to_builtin %0: !pop.scalar<ty> to i8
  kgen.return
}

// This has a single viable implementation.
// CHECK-LABEL: kgen.func @use_Itf3() {
// CHECK-NEXT:    kgen.call @"genItf3_impl0,ty=f32"()
kgen.generator @use_Itf3() {
  kgen.call @genItf3<ty: dtype = f32>() : () -> ()
  kgen.return
}

//===----------------------------------------------------------------------===//

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


//===----------------------------------------------------------------------===//

// Test that parameter and result argument types get rewritten and specialized.

// CHECK-LABEL: kgen.func @"float_constant_f32,value=1.5,type=f32"() -> !pop.scalar<f32> {
// ...
// CHECK:    %[[V1:.*]] = llvm.fptrunc
// CHECK:    %[[V2:.*]] = pop.cast_from_builtin %[[V1]] : f32 to !pop.scalar<f32>
// CHECK:    kgen.return %[[V2]] : !pop.scalar<f32>

kgen.generator @float_constant_f32<value: f64, type: dtype>() -> !pop.scalar<type>
  constraints <[eq(:dtype type, f32), "float please"]>  {
  %0 = kgen.param.constant: f64 = <value>
  %1 = llvm.fptrunc %0 : f64 to f32
  %2 = pop.cast_from_builtin %1: f32 to !pop.scalar<type>
  kgen.return %2 : !pop.scalar<type>
}

// CHECK-LABEL: kgen.func @test_f32() -> f32 {
// CHECK:    %[[V0:.*]] = kgen.call @"float_constant_f32,value=1.5,type=f32"() : () -> !pop.scalar<f32>
// CHECK:    %[[V1:.*]] = pop.cast_to_builtin %[[V0]] : !pop.scalar<f32> to f32
kgen.generator @test_f32() -> f32 {
  kgen.param.declare type : dtype = <f32>
  %1 = kgen.call @float_constant_f32<value: f64 = 1.5, type: dtype = type>() : () -> !pop.scalar<type>
  %2 = pop.cast_to_builtin %1 : !pop.scalar<type> to f32
  kgen.return %2 : f32
}

//===----------------------------------------------------------------------===//

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

  // CHECK-NEXT:   "impl0"() : () -> !pop.scalar<ui32>
  "impl0"() : () -> !kgen.paramref<ty1>

  // CHECK-NEXT: = kgen.call @"parametricAdd,sz=1,dt=ui64"
  // CHECK-SAME: (%[[ARG0:.*]], %[[ARG0:.*]]) : (!pop.scalar<ui64>, !pop.scalar<ui64>) -> !pop.scalar<ui64>
  %0 = kgen.call @parametricAdd<sz=1, dt: dtype = ui64>(%arg0, %arg0) : (!pop.scalar<ui64>, !pop.scalar<ui64>) -> !pop.scalar<ui64>

  // CHECK-NEXT: = kgen.call @"parametricAdd,sz=2,dt=f32"(%[[ARG1]], %[[ARG1]]) : (!pop.simd<2, f32>, !pop.simd<2, f32>) -> !pop.simd<2, f32>
  %1 = kgen.call @parametricAdd<sz=2, dt: dtype = f32>(%arg1, %arg1) : (!pop.simd<2, f32>, !pop.simd<2, f32>) -> !pop.simd<2, f32>

  kgen.return
}

// CHECK-LABEL: kgen.func @"takeUnary,dt=si32,fn=doubleExample"()
// CHECK: %0 = kgen.call @"doubleExample,dt=si32"(%cst) : (!pop.scalar<si32>) -> !pop.scalar<si32>
// CHECK:  %1 = kgen.call @"doubleExample,dt=si32"(%0) : (!pop.scalar<si32>) -> !pop.scalar<si32>

// CHECK-LABEL: kgen.func @"takeUnary,dt=f32,fn=nopExample"() {
// CHECK:    %0 = kgen.call @"nopExample,dt=f32"(%cst) : (!pop.scalar<f32>) -> !pop.scalar<f32>
// CHECK:    %1 = kgen.call @"nopExample,dt=f32"(%0) : (!pop.scalar<f32>) -> !pop.scalar<f32>


// CHECK-LABEL: kgen.func @"takeUnary,dt=si32,fn=test_region_concrete_region_2"() {
// CHECK:    %cst = pop.constant(#M.dense_array<1> : vector<1xsi32>) : !pop.scalar<si32>
// CHECK:    %0 = pop.add %cst, %cst : !pop.scalar<si32>
// CHECK:    %1 = pop.mul %0, %cst : !pop.scalar<si32>
// CHECK:    %2 = pop.add %1, %1 : !pop.scalar<si32>
// CHECK:    %3 = pop.mul %2, %1 : !pop.scalar<si32>

kgen.generator @takeUnary
  <dt: dtype, fn: <dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt>>() {

  %0 = pop.constant(1) : !pop.scalar<dt>
  %1 = kgen.call_param[<dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt>: fn]<dt: dtype = dt>(%0)
  %2 = kgen.call_param[<dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt>: fn]<dt: dtype = dt>(%1)
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
   fn: <sz, dt: dtype>(!pop.simd<sz,dt>, !pop.simd<sz,dt>) -> !pop.simd<sz,dt>
  >() {

  %0 = pop.constant(1) : !pop.scalar<dt>

  %1 = kgen.call_param[<sz, dt: dtype>(!pop.simd<sz,dt>, !pop.simd<sz,dt>) -> !pop.simd<sz,dt>: fn]
                <sz=1, dt: dtype = dt>(%0, %0)
  kgen.return
}

// CHECK-LABEL:  kgen.func @test_symbol() {
kgen.generator @test_symbol() {
  // CHECK: kgen.call @"takeUnary,dt=si32,fn=doubleExample"()
  kgen.call @takeUnary<dt: dtype = si32,
     fn : <dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt> = @doubleExample>() : () -> ()

  // CHECK: kgen.call @"takeUnary,dt=f32,fn=nopExample"()
  kgen.call @takeUnary<dt: dtype = f32,
     fn : <dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt> = @nopExample>() : () -> ()

  // CHECK: kgen.call @"takeParametricBinary,sz=2,dt=f32,fn=parametricAdd"()
  kgen.call @takeParametricBinary
     <
      sz = 2,
      dt: dtype = f32,
      fn : <sz, dt: dtype>(!pop.simd<sz,dt>, !pop.simd<sz,dt>) -> !pop.simd<sz,dt> = @parametricAdd
     >() : () -> ()

  kgen.return
}

// This function is instantiated with regions defined below.
kgen.generator @take_non_parametric_f32<fn: (!pop.scalar<f32>) -> !pop.scalar<f32>>() {
  %0 = pop.constant(1.0:f32) : !pop.scalar<f32>
  %1 = kgen.call_param[(!pop.scalar<f32>) -> !pop.scalar<f32>: fn](%0)
  %2 = kgen.call_param[(!pop.scalar<f32>) -> !pop.scalar<f32>: fn](%1)
  kgen.return
}

// CHECK-LABEL: kgen.func @"take_non_parametric_f32,fn=test_region_concrete_region_0"() {
// CHECK:   %cst = pop.constant(1.000000e+00 : f32) : !pop.scalar<f32>
// CHECK:   %0 = pop.mul %cst, %cst : !pop.scalar<f32>
// CHECK:   %1 = pop.mul %0, %0 : !pop.scalar<f32>
// CHECK:   kgen.return
// CHECK-LABEL: kgen.func @"take_non_parametric_f32,fn=test_region_concrete_region_1"() {
// CHECK:   %cst = pop.constant(1.000000e+00 : f32) : !pop.scalar<f32>
// CHECK:   %0 = pop.add %cst, %cst : !pop.scalar<f32>
// CHECK:   %1 = pop.add %0, %0 : !pop.scalar<f32>
// CHECK:   kgen.return

// CHECK-LABEL:  kgen.func @test_region() {
kgen.generator @test_region() {
  // CHECK:  kgen.call @"take_non_parametric_f32,fn=test_region_concrete_region_0"() : () -> ()
  kgen.call @take_non_parametric_f32<
    fn : (!pop.scalar<f32>) -> !pop.scalar<f32> = region>() : () -> ()
    fn(%arg0: !pop.scalar<f32>) {
      %result = pop.mul %arg0, %arg0 : !pop.scalar<f32>
      kgen.return %result : !pop.scalar<f32>
    }

  // CHECK: kgen.call @"take_non_parametric_f32,fn=test_region_concrete_region_1"()

  // This is the same as above, but calling through a parameter.  This shows the
  // kgen.call_param -> kgen.call lowering maintains the region correctly.
  kgen.param.declare take_non_parametric_f32
    : <fn: (!pop.scalar<f32>) -> !pop.scalar<f32>>()->() = <@take_non_parametric_f32>
  kgen.call_param[<fn: (!pop.scalar<f32>) -> !pop.scalar<f32>>()->(): take_non_parametric_f32]
    <fn : (!pop.scalar<f32>) -> !pop.scalar<f32> = region>()
    fn(%arg0: !pop.scalar<f32>) {
      %result = pop.add %arg0, %arg0 : !pop.scalar<f32>
      kgen.return %result : !pop.scalar<f32>
    }

  // Check a call to a parametric region.
  // CHECK: kgen.call @"takeUnary,dt=si32,fn=test_region_concrete_region_2"()
  kgen.call @takeUnary<dt: dtype = si32,
     fn : <dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt> = region>() : () -> ()
    fn<dt:dtype>(%arg0: !pop.scalar<dt>) {
      %0 = pop.add %arg0, %arg0 : !pop.scalar<dt>
      %1 = pop.mul %0, %arg0 : !pop.scalar<dt>
      kgen.return %1 : !pop.scalar<dt>
    }

  kgen.return
}

// CHECK:  kgen.func @"just_call_it_pass_it,fn=test_region_insanity_concrete_region_0,littleFn=test_region_insanity_concrete_region_1"() {
// CHECK:    %cst = pop.constant(#M.dense_array<1.000000e+00> : vector<1xf64>) : !pop.scalar<f64>
// CHECK:    %0 = kgen.param.constant  = <127>
// CHECK:    kgen.return
kgen.generator @just_call_it_pass_it
  <fn: <subFn:<dt: dtype->index>()->()>()->(),
   littleFn: <dt: dtype->index>()->()>() {

  kgen.call_param[<subFn : <dt: dtype->index>()->()>()->(): fn]<subFn: <dt: dtype->index>()->() = littleFn>()
  kgen.return
}

// CHECK-LABEL: @test_region_insanity
kgen.generator @test_region_insanity() {
  // CHECK: kgen.call @"just_call_it_pass_it,fn=test_region_insanity_concrete_region_0,littleFn=test_region_insanity_concrete_region_1"()
  kgen.call @just_call_it_pass_it
          <fn: <subFn:<dt: dtype->index>()->()>()->() = region, littleFn: <dt: dtype->index>()->() = region>() : () -> ()
    fn<subFn:<dt: dtype->index>()->()>() {
      kgen.call_param[<dt: dtype->index>()->(): subFn]<dt: dtype = f64->resultParam>()
      %0 = kgen.param.constant = <add(resultParam, 4)>
      kgen.return
    },
    littleFn<dt: dtype->index>() {
      %0 = pop.constant(1) : !pop.scalar<dt>
      kgen.return<123>
    }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @"parametricBinOp,ty=!pop.scalar<f32>"
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>, %[[ARG1:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
// CHECK-NEXT: %[[V0:.*]] = "custom_op"(%[[ARG0]], %[[ARG1]]) : (!pop.scalar<f32>, !pop.scalar<f32>) -> !pop.scalar<f32>
// CHECK-NEXT: kgen.return %[[V0]] : !pop.scalar<f32>
kgen.generator @parametricBinOp<ty: type>
  (%a: !kgen.paramref<ty>, %b: !kgen.paramref<ty>) -> !kgen.paramref<ty> {
  %res = "custom_op" (%a, %b) : (!kgen.paramref<ty>, !kgen.paramref<ty>) -> !kgen.paramref<ty>
  kgen.return %res : !kgen.paramref<ty>
}

// CHECK-LABEL: kgen.func @"takeParametricBinary,dt=f32,fn=parametricBinOp"() {
kgen.generator @takeParametricBinary
  <dt: dtype,
   fn: <ty: type>(!kgen.paramref<ty>, !kgen.paramref<ty>) -> !kgen.paramref<ty>
  >() {

  %0 = pop.constant(1) : !pop.scalar<dt>

  // CHECK: kgen.call @"parametricBinOp,ty=!pop.scalar<f32>"
  %1 = kgen.call_param[<ty: type>(!kgen.paramref<ty>, !kgen.paramref<ty>) -> !kgen.paramref<ty>: fn]
                <ty: type = !pop.scalar<dt>>(%0, %0)
  kgen.return
}

// CHECK-LABEL: kgen.func @test_paramref_type_rewrite() {
kgen.generator @test_paramref_type_rewrite() {
  // CHECK: kgen.call @"takeParametricBinary,dt=f32,fn=parametricBinOp"() : () -> ()
  kgen.call @takeParametricBinary<dt: dtype = f32,
      fn : <ty: type>(!kgen.paramref<ty>, !kgen.paramref<ty>) -> !kgen.paramref<ty>
      = @parametricBinOp>() : () -> ()

  kgen.return
}

// -----

kgen.generator.interface @foo<size>(index) -> index

kgen.generator @foo1<size>(%a: index) -> index
    constraints <[eq(size, 1), "1"]>
    implements @foo {
  kgen.return %a : index
}

kgen.generator @foo2<size>(%a: index) -> index
    constraints <[eq(size, 2), "2"]>
    implements @foo {
  kgen.return %a : index
}

kgen.generator @bar<T:type>(%a: !kgen.paramref<T>) -> !kgen.paramref<T> {
  kgen.return %a : !kgen.paramref<T>
}

kgen.generator @baz<() -> index>() {
  kgen.return<50>
}

// CHECK-LABEL: kgen.func @parametric_addressof
kgen.generator @parametric_addressof() {
  // CHECK-NEXT: kgen.addressof @"foo1,size=1" : (index) -> index
  %0 = kgen.addressof @foo<size = 1> : (index) -> index
  // CHECK-NEXT: kgen.addressof @"foo2,size=2" : (index) -> index
  %1 = kgen.addressof @foo<size = 2> : (index) -> index
  // CHECK-NEXT: kgen.addressof @"bar,T=i32" : (i32) -> i32
  %2 = kgen.addressof @bar<T:type = i32> : (i32) -> i32
  // CHECK-NEXT: kgen.addressof @baz<() -> result> : () -> ()
  %3 = kgen.addressof @baz<() -> result> : () -> ()
  kgen.return
}

// -----

kgen.struct.decl @Int20 {
  value : i20
}

kgen.struct.decl @Int40 {
  value : i40
}

kgen.struct.decl @Pair<T1: type, T2: type> {
  first : !kgen.paramref<T1>
  second : !kgen.paramref<T2>
}

// CHECK-LABEL: @"struct_sizeof
kgen.generator @struct_sizeof<T1: type, T2: type>() {
  // CHECK-NEXT: <4>
  %0 = kgen.param.constant = <get_alignof(!kgen.ref<@Int20>)>
  // CHECK-NEXT: <4>
  %1 = kgen.param.constant = <get_sizeof(!kgen.ref<@Int20>)>
  // CHECK-NEXT: <8>
  %2 = kgen.param.constant = <get_alignof(!kgen.ref<@Pair<T1: type = T1, T2: type = T2>>)>
  // CHECK-NEXT: <16>
  %3 = kgen.param.constant = <get_sizeof(!kgen.ref<@Pair<T1: type = T1, T2: type = T2>>)>
  kgen.return
}

kgen.generator @elaborate() {
  kgen.call @struct_sizeof<T1: type = !kgen.ref<@Int40>, T2: type = !kgen.ref<@Int20>>() : () -> ()
  kgen.return
}

// -----

// This takes a parameter function that uses a contextual type instead of
// to-be-bound types.
// CHECK-LABEL: kgen.func @"takeFnContextualType,ty=index,fn=sillyFn"() -> index {
// CHECK:  %0 = kgen.call @sillyFn() : () -> index
kgen.generator @takeFnContextualType<ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> {
  %0 = kgen.call_param[()->!kgen.paramref<ty>: fn]()
  kgen.return %0: !kgen.paramref<ty>
}

kgen.func @sillyFn() -> index {
  %0 = kgen.param.constant = <42>
  kgen.return %0: index
}

// CHECK-LABEL:  kgen.func @elaborateFnWithContextualType() -> index {
// CHECK:   %0 = kgen.call @"takeFnContextualType,ty=index,fn=sillyFn"() : () -> index
kgen.generator @elaborateFnWithContextualType() -> index {
  %0 = kgen.call @takeFnContextualType<ty: type = index, fn: ()->index = @sillyFn>() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: @elaborateFnWithContextualType2()
kgen.generator @elaborateFnWithContextualType2() -> (index, index) {
  // Show we can bind a generic signature to a concrete one.
  kgen.param.declare boundFn: ()->index =
    <bind_signature(:<ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> @takeFnContextualType,
                    :type index, :()->index @sillyFn)>

  // CHECK-NEXT: %0 = kgen.call @"takeFnContextualType,ty=index,fn=sillyFn"()
  %0 = kgen.call_param[()->index: boundFn]()

  kgen.param.declare fn: <ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> = <@takeFnContextualType>

  kgen.param.declare boundFn2: ()->index =
    <bind_signature(:<ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> fn,
                    :type index, :()->index @sillyFn)>

  // CHECK-NEXT: %1 = kgen.call @"takeFnContextualType,ty=index,fn=sillyFn"()
  %1 = kgen.call_param[()->index: boundFn2]()

  kgen.return %0, %1 : index, index
}

// -----

// CHECK-LABEL: kgen.func @top
kgen.generator @top() {
  // CHECK: kgen.call @"mid,fn=top_concrete_region_0,N=4"()
  %0:2 = kgen.call @mid<fn: <fn: ()->index>() -> index = region, N=4>() : () -> (index, index)
  fn<fn: ()->index>() {
    %0 = kgen.call_param[()->index: fn]()
    kgen.return %0 : index
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @"mid,fn=top_concrete_region_0,N=4"
kgen.generator @mid<fn: <fn: ()->index>() -> index, N>() -> (index, index) {
  // CHECK: %[[C0:.*]] = index.constant 0
  // CHECK: %[[C1:.*]] = index.constant 1
  // CHECK: %[[C4:.*]] = kgen.param.constant = <4>
  // CHECK: %[[ADD:.*]] = index.add %[[C1]], %[[C4]]
  // CHECK: return %[[C0]], %[[ADD]]
  %c0 = index.constant 0
  %c1 = index.constant 1
  %1 = kgen.inlined_call[<fn: ()->index>() -> index: fn]<fn: ()->index = region>()
  fn() {
    kgen.return %c0 : index
  }
  %2 = kgen.inlined_call[<fn: ()->index>() -> index: fn]<fn: ()->index = region>()
  fn() {
    %3 = kgen.param.constant = <N>
    %5 = index.add %c1, %3
    kgen.return %5 : index
  }
  kgen.return %1, %2 : index, index
}

// -----

// CHECK-LABEL: kgen.func @outermost
kgen.generator @outermost() -> index{
  // CHECK: kgen.call @"middle,outer=outermost_concrete_region_0"
  %1 = kgen.call @middle<outer:<fn:()->index>()->index = region>() : () -> index
  outer<fn:()->index>() {
    %2 = kgen.call_param[()->index:fn]()
    kgen.return %2 : index
  }
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func @"middle,outer=outermost_concrete_region_0"
kgen.generator @middle<outer:<fn:()->index>()->index>() -> index{
  // CHECK: %[[X:.*]] = index.constant 1
  %x = index.constant 1
  %1 = kgen.inlined_call[<fn:()->index, outer:<fn:()->index>()->index>()->index : @inner]
                        <fn: () -> index = region, outer:<fn:()->index>()->index = outer>()
  fn() {
    kgen.return %x : index
  }
  // CHECK: return %[[X]]
  kgen.return %1 : index
}

// COM: Inlined instations of symbols get removed.
// CHECK-NOT: kgen.func @"inner
kgen.generator @inner<fn: ()->index, outer:<fn:()->index>()->index>() -> index {
  %0 = kgen.call_param[<fn:()->index>()->index: outer]<fn:()->index=fn>()
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: @inlined_region_return
kgen.generator @inlined_region_return() {
  // CHECK: kgen.param.constant = <2>
  kgen.inlined_call[<fn: <()->index>()->()>()->(): @wants_region_return]<fn: <()->index>()->() = region>()
  fn<()->index>() {
    kgen.return<2>
  }
  kgen.return
}

// CHECK-NOT: @wants_region_return
kgen.generator @wants_region_return<fn: <()->index>()->()>() {
  kgen.call_param[<()->index>()->(): fn]<() -> result>()
  %0 = kgen.param.constant = <result>
  kgen.return
}

// -----

kgen.generator.interface @iface() -> index

// CHECK-LABEL: kgen.func @iface1
kgen.generator @iface1() -> index implements @iface {
  %0 = index.constant 0
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @iface2
kgen.generator @iface2() -> index implements @iface {
  %0 = index.constant 1
  kgen.return %0 : index
}

// CHECK-NOT @"two_instances
kgen.generator @two_instances<fn: ()->index>() -> index {
  %0 = kgen.call @iface() : () -> index
  %1 = kgen.call_param[()->index: fn]()
  %2 = index.add %0, %1
  kgen.return %2 : index
}

// CHECK-LABEL: kgen.func @inline_call_two_instances
kgen.generator @inline_call_two_instances(%arg0: index) -> index {
  // CHECK-NEXT: kgen.call @iface1
  // CHECK-NEXT: index.add %0, %arg0
  %0 = kgen.inlined_call[<fn: ()->index>() -> index: @two_instances]<fn: ()->index = region>()
  fn() {
    kgen.return %arg0 : index
  }
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @inline_call_two_instances_concrete_1
// CHECK-NEXT: kgen.call @iface2
// CHECK-NEXT: index.add %0, %arg0

// -----

kgen.generator.interface @iface<fn: ()->index>() -> index

kgen.generator @iface1<fn: ()->index>() -> index implements @iface {
  %0 = index.constant 0
  %1 = kgen.call_param[()->index: fn]()
  %2 = index.add %0, %1
  kgen.return %2 : index
}

kgen.generator @iface2<fn: ()->index>() -> index implements @iface {
  %0 = index.constant 1
  %1 = kgen.call_param[()->index: fn]()
  %2 = index.add %0, %1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @inline_call_interface
kgen.generator @inline_call_interface(%arg0: index) -> index {
  // CHECK: index.add %0, %arg0
  %0 = kgen.inlined_call[<fn: ()->index>()->index: @iface]<fn: ()->index = region>()
  fn() {
    kgen.return %arg0: index
  }
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @inline_call_interface_concrete_0
// CHECK: index.add %0, %arg0

// -----

// CHECK-LABEL: kgen.func @"invokeWithN,N=1
// CHECK-NEXT: constant = <11>

// CHECK-LABEL: kgen.func @"invokeWithN,N=2
// CHECK-NEXT: constant = <20>

kgen.generator @invokeWithN<N, fn: <N>() -> index>() -> index{
  %0 = kgen.call_param[<N>() -> index: fn]<N = N>()
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @"invokeTwice,M=10"
// CHECK-NEXT: kgen.call @"invokeWithN,N=1
// CHECK-NEXT: kgen.call @"invokeWithN,N=2
kgen.generator @invokeTwice<M>() {
  %0 = kgen.call @invokeWithN<N = 1, fn: <N>() -> index = region>() : () -> index
  fn<N>() {
    %1 = kgen.param.constant = <add(N, M)>
    kgen.return %1 : index
  }
  %1 = kgen.call @invokeWithN<N = 2, fn: <N>() -> index = region>() : () -> index
  fn<N>() {
    %1 = kgen.param.constant = <mul(N, M)>
    kgen.return %1 : index
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @doIt
// CHECK-NEXT: kgen.call @"invokeTwice,M=10"
kgen.generator @doIt() {
  kgen.call @invokeTwice<M = 10>() : () -> ()
  kgen.return
}

// -----

// This test case caught a tricky bug with name shadowing.

// CHECK-LABEL: kgen.func @"invokeWithN,N=1
// CHECK-NEXT: constant = <11>

// CHECK-LABEL: kgen.func @"invokeWithN,N=2
// CHECK-NEXT: constant = <20>

kgen.generator @invokeWithN<N, fn: <N>() -> index>() -> index{
  %0 = kgen.call_param[<N>() -> index: fn]<N = N>()
  kgen.return %0 : index
}

kgen.generator @aliasN<N>() {
  kgen.param.declare M = <N>
  %0 = kgen.call @invokeWithN<N = 1, fn: <N>() -> index = region>() : () -> index
  fn<N>() {
    %1 = kgen.param.constant = <add(N, M)>
    kgen.return %1 : index
  }
  %1 = kgen.call @invokeWithN<N = 2, fn: <N>() -> index = region>() : () -> index
  fn<N>() {
    %1 = kgen.param.constant = <mul(N, M)>
    kgen.return %1 : index
  }
  kgen.return
}

kgen.generator @doIt() {
  kgen.call @aliasN<N = 10>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @"nestMe,fn=nestMe,fn=nestMe,fn=tripleNested,A=1_region_0_region_0_region_0"
// CHECK-NEXT: kgen.param.constant = <6>
kgen.generator @nestMe<fn: () -> index>() -> index {
  %0 = kgen.call_param[() -> index: fn]()
  kgen.return %0 : index
}

kgen.generator @tripleNested<A>() -> index{
  %0 = kgen.call @nestMe<fn: () -> index = region>() : () -> index
  fn() {
    kgen.param.declare B = <2>
    %1 = kgen.call @nestMe<fn: () -> index = region>() : () -> index
    fn() {
      kgen.param.declare C = <3>
      %2 = kgen.call @nestMe<fn: () -> index = region>() : () -> index
      fn() {
        %3 = kgen.param.constant = <add(A, B, C)>
        kgen.return %3 : index
      }
      kgen.return %2 : index
    }
    kgen.return %1 : index
  }
  kgen.return %0 : index
}

kgen.generator @doIt() {
  %0 = kgen.call @tripleNested<A=1>() : () -> index
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @"nestMe,N=6,fn=nestMe,N=4,fn=nestMe,N=2,fn=tripleNested,A=1_region_0_region_0_region_0"
// CHECK-NEXT: kgen.param.constant = <21>

kgen.generator @nestMe<N, fn: <N>() -> index>() -> index {
  %0 = kgen.call_param[<N>() -> index: fn]<N=N>()
  kgen.return %0 : index
}

kgen.generator @tripleNested<A>() -> index{
  %0 = kgen.call @nestMe<N = 2, fn: <N>() -> index = region>() : () -> index
  fn<N>() {
    kgen.param.declare B = <N>
    kgen.param.declare C = <3>
    %1 = kgen.call @nestMe<N = 4, fn: <N>() -> index = region>() : () -> index
    fn<N>() {
      kgen.param.declare D = <N>
      kgen.param.declare E = <5>
      %2 = kgen.call @nestMe<N = 6, fn: <N>() -> index = region>() : () -> index
      fn<N>() {
      kgen.param.declare F = <N>
        %3 = kgen.param.constant = <add(A, B, C, D, E, F)>
        kgen.return %3 : index
      }
      kgen.return %2 : index
    }
    kgen.return %1 : index
  }
  kgen.return %0 : index
}

kgen.generator @doIt() {
  %0 = kgen.call @tripleNested<A=1>() : () -> index
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @passArgument
kgen.generator @passArgument(%arg0: index) -> index {
  // CHECK: return %arg0
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.func @doIt
kgen.generator @doIt() -> index {
  // CHECK: %0 = index.constant 2
  %0 = index.constant 2
  %1 = kgen.inlined_call[(index) -> index: @passArgument](%0)
  // CHECK: return %0 : index
  kgen.return %0 : index
}

/// This evaluator returns a constant index 0.
kgen.generator @first<FN:type>(%funcs: !pop.pointer<FN>, %size: index) -> index {
  %0 = kgen.param.constant = <0>
  kgen.return %0 : index
}

/// Always pick the second implementation of this interface (because of defaultImpl).
kgen.generator.interface @pickFirst()
  evaluator (!pop.pointer<() -> ()>, index) -> index = @first<FN:type=()->()>
  defaultImpl () -> () = @pickFirstB

kgen.generator @pickFirstA() implements @pickFirst {
  kgen.return
}

kgen.generator @pickFirstB() implements @pickFirst {
  kgen.return
}

// CHECK-LABEL: @test
kgen.generator @test() {
  // CHECK-NEXT: kgen.call @pickFirstB
  kgen.call @pickFirst() : () -> ()
  kgen.return
}
