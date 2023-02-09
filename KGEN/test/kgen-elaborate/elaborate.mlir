// RUN: kgen-opt %s -split-input-file -elaborate-generators="enable-search=true" -allow-unregistered-dialect | FileCheck %s

// This is left untouched.
// CHECK-LABEL: kgen.func @test0() -> index {
// CHECK-NEXT: %[[V0:.*]] = kgen.param.constant = <1>
// CHECK-NEXT:  kgen.return %[[V0]] : index
// CHECK-NEXT: }
kgen.generator @test0<() -> result>() -> index {
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

kgen.generator @genA<size, type: dtype, val: f32 -> result: index>(%arg0: si32) -> si32 {

  %0 = kgen.param.constant = <add(size, 4)>
  %1 = kgen.param.constant: dtype = <type>
  %2 = kgen.param.constant: f32 = <val>

  // Silly op so we know when something used this.
  "genA.op"() { value = #kgen.param.decl.ref<"size"> : index} : () -> !pop.scalar<type>

  kgen.return<mul(size, 2)> %arg0 : si32
}
// CHECK-LABEL: kgen.func @"genA,size=42,type=f32,val=2"
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:   %[[V0:.*]] = kgen.param.constant  = <46>
// CHECK-NEXT:   %[[V1:.*]] = kgen.param.constant: dtype = <f32>
// CHECK-NEXT:   %[[V2:.*]] = kgen.param.constant: f32 = <2.000000e+00>
// CHECK-NEXT:   %[[V3:.*]] = "genA.op"() {value = 42 : index} : () -> !pop.scalar<f32>
// CHECK-NEXT:   kgen.return %[[ARG0]] : si32
// CHECK-NEXT: }

// CHECK-LABEL: kgen.func @"genA,size=19,type=si8,val=1.5"
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:    %[[V0:.*]] = kgen.param.constant  = <23>
// CHECK-NEXT:    %[[V1:.*]] = kgen.param.constant: dtype = <si8>
// CHECK-NEXT:    %[[V2:.*]] = kgen.param.constant: f32 = <1.500000e+00>
// CHECK-NEXT:    %[[V3:.*]] = "genA.op"() {value = 19 : index} : () -> !pop.scalar<si8>
// CHECK-NEXT:    kgen.return %[[ARG0]] : si32
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
  %1 = kgen.call @genA<size = our_size, type : dtype = f32, val : f32 = 2.0 -> resultSizeA = result>(%arg0) : (si32) -> si32
  // CHECK-NEXT: %{{.*}} = kgen.call @"genA,size=42,type=f32,val=2"(%[[ARG0]]) : (si32) -> si32

  %2 = kgen.call @genA<size = 19, type : dtype = si8, val : f32 = 1.5 -> resultSizeB = result>(%arg1) : (si32) -> si32
  // CHECK-NEXT: %{{.*}} = kgen.call @"genA,size=19,type=si8,val=1.5"(%[[ARG1]]) : (si32) -> si32

  %3 = kgen.call @genA<size = 19, type : dtype = si8, val : f32 = 1.5 -> resultSizeC = result>(%arg1) : (si32) -> si32
  // CHECK-NEXT: %{{.*}} = kgen.call @"genA,size=19,type=si8,val=1.5"(%[[ARG1]]) : (si32) -> si32


  %4 = kgen.param.constant = <resultSizeA>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <84>

  %5 = kgen.param.constant = <resultSizeB>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <38>

  %6 = kgen.param.constant = <resultSizeC>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <38>

  %7 = kgen.call @test0<() -> kernelResult = result>() : () -> index
  // CHECK-NEXT: %{{.*}} = kgen.call @test0()

  %8 = kgen.param.constant = <kernelResult>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <123456>

  kgen.return %0, %1, %2, %4, %5 : si32, si32, si32, index, index
}

//===----------------------------------------------------------------------===//

// CHECK-NOT: kgen.generator.interface @genItf
kgen.generator.interface @genItf<x -> result>(si32) -> si32

// CHECK-LABEL: kgen.func @"genItf_impl1,x=42"
// CHECK-SAME: %[[ARG0:.*]]: si32
// CHECK-NEXT:   "genItf.impl1"() {value = 42 : index} : () -> ()
// CHECK-NEXT:   kgen.return %[[ARG0]] : si32
// CHECK-NEXT: }
kgen.generator @genItf_impl1<x -> result>(%arg0: si32) -> si32
  implements @genItf {
  "genItf.impl1"() { value = #kgen.param.decl.ref<"x"> : index} : () -> ()
  kgen.return<add(x, 1)> %arg0 : si32
}

// CHECK-LABEL: kgen.func @"genItf_impl2,x=42"
// CHECK-SAME: %[[ARG0:.*]]: si32
// CHECK-NEXT:   "genItf.impl2"() {value = 42 : index} : () -> ()
// CHECK-NEXT:   kgen.return %[[ARG0]] : si32
// CHECK-NEXT: }
kgen.generator @genItf_impl2<x -> result>(%arg0: si32) -> si32
  implements @genItf {
  "genItf.impl2"() { value = #kgen.param.decl.ref<"x"> : index} : () -> ()
  kgen.return<mul(x, 2)> %arg0 : si32
}

// CHECK-LABEL: kgen.func @use_interface(
// CHECK-SAME: %[[ARG0:.*]]: si32
// CHECK-NEXT: %[[V0:.*]] = kgen.call @"genItf_impl1,x=42"(%[[ARG0]])
// CHECK-NEXT: %[[V1:.*]] = kgen.param.constant = <43>

// CHECK-LABEL: kgen.func @use_interface_concrete_0
// CHECK-SAME: %[[ARG0:.*]]: si32
// CHECK-NEXT:    %{{.*}} = kgen.call @"genItf_impl2,x=42"(%[[ARG0]]) : (si32) -> si32
// CHECK-NEXT:     %{{.*}} = kgen.param.constant = <84>
kgen.generator @use_interface(%arg0: si32) -> index {
  %0 = kgen.call @genItf<x = 42 -> out = result>(%arg0) : (si32) -> si32
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

// CHECK-LABEL: @"genItf2,x=0_2"()
kgen.generator @genItf2<x>() {
  // CHECK-NEXT: kgen.call @"genItf2_impl0,x=0"
  kgen.param.search impl : () -> () = <@genItf2_impl0<x = x>, @genItf2_impl1<x = x>>
  kgen.call_param[() -> () : impl]()
  kgen.return
}

// CHECK-NOT: kgen.func @"genItf2_impl0,x=1"() {
// CHECK-LABEL: kgen.func @"genItf2_impl0,x=0"() {
// CHECK-NEXT:   "impl.0"() : () -> ()
// CHECK-NEXT:   kgen.return
// CHECK-NOT: kgen.func @"genItf2_impl0,x=1"() {
kgen.generator @genItf2_impl0<x>()
  constraints <[eq(x, 0), "x must be zero"]> {
  "impl.0"() : () -> ()
  kgen.return
}

// CHECK-NOT: kgen.func @"genItf2_impl1,x=0"()
// CHECK-LABEL: kgen.func @"genItf2_impl1,x=1"() {
// CHECK-NEXT:   "impl.1"() : () -> ()
// CHECK-NEXT:   kgen.return
// CHECK-NOT: kgen.func @"genItf2_impl1,x=0"()
kgen.generator @genItf2_impl1<x>()
  constraints <[eq(x, 1), "x must be 1"]> {
  "impl.1"() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @use_Itf2zero() {
// CHECK-NEXT:   kgen.call @"genItf2,x=0_2"() : () -> ()
// CHECK-NEXT:   kgen.return
kgen.generator @use_Itf2zero() {
  kgen.call @genItf2<x = 0>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @use_Itf2one() {
// CHECK-NEXT:   kgen.call @"genItf2,x=1_5"() : () -> ()
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
  "impl.0"() : () -> ()
  kgen.return
}

// This generates a kernel that fails to verify, so it isn't used and must be
// deleted.
// CHECK-NOT: genItf3_impl1
kgen.generator @genItf3_impl1<ty: dtype>() implements @genItf3 {
  %one = kgen.param.constant: scalar<si64> = <1>
  %0 = pop.cast %one : !pop.scalar<si64> to !pop.scalar<ty>
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
// CHECK-NEXT: kgen.call @"genItf_impl1,x=42"(%[[ARG0]]) : (si32) -> si32
// CHECK-NEXT: kgen.call @"genItf_impl1,x=42"(%[[ARG0]]) : (si32) -> si32
// CHECK-NEXT: kgen.call @use_interface(%[[ARG0]])

// CHECK-NOT: kgen.func @track_expansions

// CHECK-LABEL: kgen.func @track_expansions_concrete_6
// CHECK-SAME: (%[[ARG0:.*]]: si32)
// CHECK-NEXT: kgen.call @"genItf_impl2,x=42"
// CHECK-NEXT: kgen.call @"genItf_impl2,x=42"
// CHECK-NEXT: kgen.call @use_interface_concrete_0(%[[ARG0]])

// CHECK-NOT: kgen.func @track_expansions

kgen.generator @track_expansions(%arg0: si32) {
  // Within any generated kernel genItf should expand the same way.
  %0 = kgen.call @genItf<x = 42 -> out = result>(%arg0) : (si32) -> si32
  %1 = kgen.call @genItf<x = 42 -> out1 = result>(%arg0) : (si32) -> si32

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

kgen.generator.interface @getSIMDLength<dt: dtype -> length>()

kgen.generator @getSIMDLengthF32<dt: dtype -> length>()
     implements @getSIMDLength {
  // This could be implemented as a constraint.
  kgen.param.assert <eq(:dtype dt, f32)>, "this only works for f32"
  // vector length for floats is 4 on our target.
  kgen.return <4>
}

kgen.generator @getSIMDLengthF64<dt: dtype -> length>()
     implements @getSIMDLength {
  // This could be implemented as a constraint.
  kgen.param.assert <eq(:dtype dt, f64)>, "this only works for f32"
  // vector length for doubles is 2 on our target.
  kgen.return <2>
}

// CHECK-LABEL: kgen.func @paramAssertExample()
// CHECK-NEXT:    kgen.call @"getSIMDLengthF32,dt=f32"()
// CHECK-NEXT:    kgen.return
kgen.generator @paramAssertExample() {
  kgen.call @getSIMDLength<dt : dtype = f32 -> flen = length>() : () -> ()

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
  %0 = kgen.call @parametricAdd<sz=1, dt: dtype = ui64>(%arg0, %arg0) : (!pop.scalar<ui64>, !pop.scalar<ui64>) -> !pop.scalar<ui64>

  // CHECK-NEXT: = kgen.call @"parametricAdd,sz=2,dt=f32"(%[[ARG1]], %[[ARG1]]) : (!pop.simd<2, f32>, !pop.simd<2, f32>) -> !pop.simd<2, f32>
  %1 = kgen.call @parametricAdd<sz=2, dt: dtype = f32>(%arg1, %arg1) : (!pop.simd<2, f32>, !pop.simd<2, f32>) -> !pop.simd<2, f32>

  kgen.return
}

// CHECK-LABEL: kgen.func @"takeUnary,dt=si32,fn=doubleExample"()
// CHECK: %0 = kgen.param.constant
// CHECK: %1 = pop.cast
// CHECK: %2 = kgen.call @"doubleExample,dt=si32"(%1) : (!pop.scalar<si32>) -> !pop.scalar<si32>
// CHECK: %3 = kgen.call @"doubleExample,dt=si32"(%2) : (!pop.scalar<si32>) -> !pop.scalar<si32>

// CHECK-LABEL: kgen.func @"takeUnary,dt=f32,fn=nopExample"() {
// CHECK:    %0 = kgen.param.constant
// CHECK:    %1 = pop.cast
// CHECK:    %2 = kgen.call @"nopExample,dt=f32"(%1) : (!pop.scalar<f32>) -> !pop.scalar<f32>
// CHECK:    %3 = kgen.call @"nopExample,dt=f32"(%2) : (!pop.scalar<f32>) -> !pop.scalar<f32>

// COM: These are disabled because we don't handle regions for now
// XCHECK-LABEL: kgen.func @"takeUnary,dt=si32,fn=test_region_concrete_region_1"() {
// XCHECK:    %0 = kgen.param.constant
// XCHECK:    %1 = pop.cast
// XCHECK:    %2 = pop.add %1, %1 : !pop.scalar<si32>
// XCHECK:    %3 = pop.mul %2, %1 : !pop.scalar<si32>
// XCHECK:    %4 = pop.add %3, %3 : !pop.scalar<si32>
// XCHECK:    %5 = pop.mul %4, %3 : !pop.scalar<si32>

kgen.generator @takeUnary
  <dt: dtype, fn: <dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt>>() {

  %one = kgen.param.constant: scalar<si64> = <1>
  %0 = pop.cast %one : !pop.scalar<si64> to !pop.scalar<dt>
  %1 = kgen.call_param[(!pop.scalar<dt>) -> !pop.scalar<dt>:
    bind_signature(:<dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt> fn, dt)](%0)
  %2 = kgen.call_param[(!pop.scalar<dt>) -> !pop.scalar<dt>:
    bind_signature(:<dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt> fn, dt)](%1)
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

  %one = kgen.param.constant: scalar<si64> = <1>
  %0 = pop.cast %one : !pop.scalar<si64> to !pop.scalar<dt>

  %1 = kgen.call_param[(!pop.scalar<dt>, !pop.scalar<dt>) -> !pop.scalar<dt>:
    bind_signature(:<sz, dt: dtype>(!pop.simd<sz,dt>, !pop.simd<sz,dt>) -> !pop.simd<sz,dt> fn, 1, dt)](%0, %0)
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

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: // This function is instantiated with regions defined below.
// COM: kgen.generator @take_non_parametric_f32<fn: (!pop.scalar<f32>) -> !pop.scalar<f32>>() {
// COM:   %0 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1.0">>
// COM:   %1 = kgen.call_param[(!pop.scalar<f32>) -> !pop.scalar<f32>: fn](%0)
// COM:   %2 = kgen.call_param[(!pop.scalar<f32>) -> !pop.scalar<f32>: fn](%1)
// COM:   kgen.return
// COM: }
// COM:
// COM: // CHECK-LABEL: kgen.func @"take_non_parametric_f32,fn=test_region_concrete_region"() {
// COM: // CHECK:   %0 = kgen.param.constant
// COM: // CHECK:   %1 = pop.mul %0, %0 : !pop.scalar<f32>
// COM: // CHECK:   %2 = pop.mul %1, %1 : !pop.scalar<f32>
// COM: // CHECK:   kgen.return
// COM: // CHECK-LABEL: kgen.func @"take_non_parametric_f32,fn=test_region_concrete_region_0"() {
// COM: // CHECK:   %0 = kgen.param.constant
// COM: // CHECK:   %1 = pop.add %0, %0 : !pop.scalar<f32>
// COM: // CHECK:   %2 = pop.add %1, %1 : !pop.scalar<f32>
// COM: // CHECK:   kgen.return
// COM:
// COM: // CHECK-LABEL:  kgen.func @test_region() {
// COM: kgen.generator @test_region() {
// COM:   // CHECK:  kgen.call @"take_non_parametric_f32,fn=test_region_concrete_region"() : () -> ()
// COM:   kgen.param.declare.region fn0 = (%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
// COM:     %result = pop.mul %arg0, %arg0 : !pop.scalar<f32>
// COM:     kgen.return %result : !pop.scalar<f32>
// COM:   }
// COM:   kgen.call @take_non_parametric_f32<
// COM:     fn : (!pop.scalar<f32>) -> !pop.scalar<f32> = fn0>() : () -> ()
// COM:
// COM:   // CHECK: kgen.call @"take_non_parametric_f32,fn=test_region_concrete_region_0"()
// COM:
// COM:   // This is the same as above, but calling through a parameter.  This shows the
// COM:   // kgen.call_param -> kgen.call lowering maintains the region correctly.
// COM:   kgen.param.declare take_non_parametric_f32
// COM:     : <fn: (!pop.scalar<f32>) -> !pop.scalar<f32>>()->() = <@take_non_parametric_f32>
// COM:   kgen.param.declare.region fn1 = (%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
// COM:     %result = pop.add %arg0, %arg0 : !pop.scalar<f32>
// COM:     kgen.return %result : !pop.scalar<f32>
// COM:   }
// COM:   kgen.call_param[()->():
// COM:     bind_signature(:<fn: (!pop.scalar<f32>) -> !pop.scalar<f32>>()->() take_non_parametric_f32, fn1)]()
// COM:
// COM:   // Check a call to a parametric region.
// COM:   // CHECK: kgen.call @"takeUnary,dt=si32,fn=test_region_concrete_region_1"()
// COM:   kgen.param.declare.region fn2 = <dt:dtype>(%arg0: !pop.scalar<dt>) -> !pop.scalar<dt> {
// COM:     %0 = pop.add %arg0, %arg0 : !pop.scalar<dt>
// COM:     %1 = pop.mul %0, %arg0 : !pop.scalar<dt>
// COM:     kgen.return %1 : !pop.scalar<dt>
// COM:   }
// COM:   kgen.call @takeUnary<dt: dtype = si32,
// COM:      fn : <dt:dtype>(!pop.scalar<dt>) -> !pop.scalar<dt> = fn2>() : () -> ()
// COM:
// COM:   kgen.return
// COM: }
// COM:
// COM: // CHECK:  kgen.func @"just_call_it_pass_it,fn=test_region_insanity_concrete_region,littleFn=test_region_insanity_concrete_region_0"() {
// COM: // CHECK:    %0 = kgen.param.constant
// COM: // CHECK:    %1 = pop.cast %0
// COM: // CHECK:    %2 = kgen.param.constant = <127>
// COM: // CHECK:    kgen.return
// COM: kgen.generator @just_call_it_pass_it
// COM:   <fn: <subFn:<dt: dtype->index>()->()>()->(),
// COM:    littleFn: <dt: dtype->index>()->()>() {
// COM:
// COM:   kgen.call_param[()->(): bind_signature(:<subFn : <dt: dtype->index>()->()>()->() fn, littleFn)]()
// COM:   kgen.return
// COM: }
// COM:
// COM: // CHECK-LABEL: @test_region_insanity
// COM: kgen.generator @test_region_insanity() {
// COM:   // CHECK: kgen.call @"just_call_it_pass_it,fn=test_region_insanity_concrete_region,littleFn=test_region_insanity_concrete_region_0"()
// COM:   kgen.param.declare.region fn = <subFn:<dt: dtype->index>()->()>() {
// COM:     kgen.call_param[<() -> index>()->(): bind_signature(:<dt: dtype->index>()->() subFn, f64)]<() ->resultParam>()
// COM:     %0 = kgen.param.constant = <add(resultParam, 4)>
// COM:     kgen.return
// COM:   }
// COM:   kgen.param.declare.region littleFn = <dt: dtype->index>() {
// COM:     %one = kgen.param.constant: !pop.scalar<si64> = <#pop.simd<1>>
// COM:     %0 = pop.cast %one : !pop.scalar<si64> to !pop.scalar<dt>
// COM:     kgen.return<123>
// COM:   }
// COM:   kgen.call @just_call_it_pass_it<
// COM:     fn: <subFn:<dt: dtype->index>()->()>()->() = fn,
// COM:     littleFn: <dt: dtype->index>()->() = littleFn>() : () -> ()
// COM:   kgen.return
// COM: }
// COM:
// -----

// CHECK-LABEL: kgen.func @"parametricBinOp,ty=!pop.scalar<f32>"
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>, %[[ARG1:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
// CHECK-NEXT: %[[V0:.*]] = "custom.op"(%[[ARG0]], %[[ARG1]]) : (!pop.scalar<f32>, !pop.scalar<f32>) -> !pop.scalar<f32>
// CHECK-NEXT: kgen.return %[[V0]] : !pop.scalar<f32>
kgen.generator @parametricBinOp<ty: type>
  (%a: !kgen.paramref<ty>, %b: !kgen.paramref<ty>) -> !kgen.paramref<ty> {
  %res = "custom.op" (%a, %b) : (!kgen.paramref<ty>, !kgen.paramref<ty>) -> !kgen.paramref<ty>
  kgen.return %res : !kgen.paramref<ty>
}

// CHECK-LABEL: kgen.func @"takeParametricBinary,dt=f32,fn=parametricBinOp"() {
kgen.generator @takeParametricBinary
  <dt: dtype,
   fn: <ty: type>(!kgen.paramref<ty>, !kgen.paramref<ty>) -> !kgen.paramref<ty>
  >() {

  %one = kgen.param.constant: scalar<si64> = <1>
  %0 = pop.cast %one : !pop.scalar<si64> to !pop.scalar<dt>

  // CHECK: kgen.call @"parametricBinOp,ty=!pop.scalar<f32>"
  %1 = kgen.call_param[(!pop.scalar<dt>, !pop.scalar<dt>) -> !pop.scalar<dt>:
    bind_signature(:<ty: type>(!kgen.paramref<ty>, !kgen.paramref<ty>) -> !kgen.paramref<ty>
      fn, !pop.scalar<dt>)](%0, %0)
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

kgen.generator @baz<() -> result>() {
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
  // CHECK-NEXT: kgen.addressof @baz : () -> ()
  %3 = kgen.addressof @baz<() -> result = result> : () -> ()
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

kgen.generator @sillyFn() -> index {
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
                    index, @sillyFn)>

  // CHECK-NEXT: %0 = kgen.call @"takeFnContextualType,ty=index,fn=sillyFn"()
  %0 = kgen.call_param[()->index: boundFn]()

  kgen.param.declare fn: <ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> = <@takeFnContextualType>

  kgen.param.declare boundFn2: ()->index =
    <bind_signature(:<ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> fn,
                    index, @sillyFn)>

  // CHECK-NEXT: %1 = kgen.call @"takeFnContextualType,ty=index,fn=sillyFn"()
  %1 = kgen.call_param[()->index: boundFn2]()

  kgen.return %0, %1 : index, index
}

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: // CHECK-LABEL: kgen.func @top
// COM: kgen.generator @top() {
// COM:   // CHECK: kgen.call @"mid,fn=top_concrete_region,N=4"()
// COM:   kgen.param.declare.region fn = <fn: ()->index>() -> index {
// COM:     %0 = kgen.call_param[()->index: fn]()
// COM:     kgen.return %0 : index
// COM:   }
// COM:   %0:2 = kgen.call @mid<fn: <fn: ()->index>() -> index = fn, N=4>() : () -> (index, index)
// COM:   kgen.return
// COM: }
// COM:
// COM: // CHECK-LABEL: kgen.func @"mid,fn=top_concrete_region,N=4"
// COM: kgen.generator @mid<fn: <fn: ()->index>() -> index, N>() -> (index, index) {
// COM:   // CHECK: %[[C0:.*]] = index.constant 0
// COM:   // CHECK: %[[C1:.*]] = index.constant 1
// COM:   // CHECK: %[[C4:.*]] = kgen.param.constant = <4>
// COM:   // CHECK: %[[ADD:.*]] = index.add %[[C1]], %[[C4]]
// COM:   // CHECK: return %[[C0]], %[[ADD]]
// COM:   %c0 = index.constant 0
// COM:   %c1 = index.constant 1
// COM:   kgen.param.declare.region fn0 = () -> index {
// COM:     kgen.return %c0 : index
// COM:   }
// COM:   %1 = kgen.call_param[()->index: bind_signature(:<fn: ()->index>() -> index fn, fn0)]()
// COM:   kgen.param.declare.region fn1 = () -> index {
// COM:     %3 = kgen.param.constant = <N>
// COM:     %5 = index.add %c1, %3
// COM:     kgen.return %5 : index
// COM:   }
// COM:   %2 = kgen.call_param[()->index: bind_signature(:<fn: ()->index>() -> index fn, fn1)]()
// COM:   kgen.return %1, %2 : index, index
// COM: }
// COM:
// COM: // -----
// COM:
// COM: // CHECK-LABEL: kgen.func @outermost
// COM: kgen.generator @outermost() -> index{
// COM:   // CHECK: kgen.call @"middle,outer=outermost_concrete_region"
// COM:   kgen.param.declare.region outer = <fn:()->index>() -> index {
// COM:     %2 = kgen.call_param[()->index:fn]()
// COM:     kgen.return %2 : index
// COM:   }
// COM:   %1 = kgen.call @middle<outer:<fn:()->index>()->index = outer>() : () -> index
// COM:   kgen.return %1 : index
// COM: }
// COM:
// COM: // CHECK-LABEL: kgen.func @"middle,outer=outermost_concrete_region"
// COM: kgen.generator @middle<outer:<fn:()->index>()->index>() -> index{
// COM:   // CHECK: %[[X:.*]] = index.constant 1
// COM:   %x = index.constant 1
// COM:   kgen.param.declare.region fn = () -> index {
// COM:     kgen.return %x : index
// COM:   }
// COM:   %1 = kgen.call @inner <fn: () -> index = fn, outer:<fn:()->index>()->index = outer>() : () -> index
// COM:   // CHECK: return %[[X]]
// COM:   kgen.return %1 : index
// COM: }
// COM:
// COM: // COM: Inlined instations of symbols get removed.
// COM: // CHECK-NOT: kgen.func @"inner
// COM: kgen.generator @inner<fn: ()->index, outer:<fn:()->index>()->index>() -> index {
// COM:   %0 = kgen.call_param[()->index: bind_signature(:<fn:()->index>()->index outer, fn)]()
// COM:   kgen.return %0 : index
// COM: }

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: // CHECK-LABEL: @inlined_region_return
// COM: kgen.generator @inlined_region_return() {
// COM:   // CHECK: kgen.param.constant = <2>
// COM:   kgen.param.declare.region fn = <()->index>() {
// COM:     kgen.return<2>
// COM:   }
// COM:   kgen.call @wants_region_return<fn: <()->index>()->() = fn>() : () -> ()
// COM:   kgen.return
// COM: }
// COM:
// COM: // CHECK-NOT: @wants_region_return
// COM: kgen.generator @wants_region_return<fn: <()->index>()->()>() {
// COM:   kgen.call_param[<()->index>()->(): fn]<() -> result>()
// COM:   %0 = kgen.param.constant = <result>
// COM:   kgen.return
// COM: }

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: kgen.generator.interface @iface() -> index
// COM:
// COM: // CHECK-LABEL: kgen.func @iface1
// COM: kgen.generator @iface1() -> index implements @iface {
// COM:   %0 = index.constant 0
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: // CHECK-LABEL: kgen.func @iface2
// COM: kgen.generator @iface2() -> index implements @iface {
// COM:   %0 = index.constant 1
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: // CHECK-NOT @"two_instances
// COM: kgen.generator @two_instances<fn: ()->index>() -> index {
// COM:   %0 = kgen.call @iface() : () -> index
// COM:   %1 = kgen.call_param[()->index: fn]()
// COM:   %2 = index.add %0, %1
// COM:   kgen.return %2 : index
// COM: }
// COM:
// COM: // CHECK-LABEL: kgen.func @inline_call_two_instances
// COM: kgen.generator @inline_call_two_instances(%arg0: index) -> index {
// COM:   // CHECK-NEXT: kgen.call @iface1
// COM:   // CHECK-NEXT: index.add %0, %arg0
// COM:   kgen.param.declare.region fn = () -> index {
// COM:     kgen.return %arg0 : index
// COM:   }
// COM:   %0 = kgen.call @two_instances<fn: ()->index = fn>() : () -> index
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: // CHECK-LABEL: kgen.func @inline_call_two_instances_concrete_1
// COM: // CHECK-NEXT: kgen.call @iface2
// COM: // CHECK-NEXT: index.add %0, %arg0

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: kgen.generator.interface @iface<fn: ()->index>() -> index
// COM:
// COM: kgen.generator @iface1<fn: ()->index>() -> index implements @iface {
// COM:   %0 = index.constant 0
// COM:   %1 = kgen.call_param[()->index: fn]()
// COM:   %2 = index.add %0, %1
// COM:   kgen.return %2 : index
// COM: }
// COM:
// COM: kgen.generator @iface2<fn: ()->index>() -> index implements @iface {
// COM:   %0 = index.constant 1
// COM:   %1 = kgen.call_param[()->index: fn]()
// COM:   %2 = index.add %0, %1
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: // CHECK-LABEL: kgen.func @inline_call_interface
// COM: kgen.generator @inline_call_interface(%arg0: index) -> index {
// COM:   // CHECK: index.add %idx0, %arg0
// COM:   kgen.param.declare.region fn = () -> index {
// COM:     kgen.return %arg0: index
// COM:   }
// COM:   %0 = kgen.call @iface<fn: ()->index = fn>() : () -> index
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: // CHECK-LABEL: kgen.func @inline_call_interface_concrete_0
// COM: // CHECK: index.add %idx1, %arg0

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: // CHECK-LABEL: kgen.func @"invokeWithN,N=1
// COM: // CHECK-NEXT: constant = <11>
// COM:
// COM: // CHECK-LABEL: kgen.func @"invokeWithN,N=2
// COM: // CHECK-NEXT: constant = <20>
// COM:
// COM: kgen.generator @invokeWithN<N, fn: <N>() -> index>() -> index{
// COM:   %0 = kgen.call_param[() -> index: bind_signature(:<N>() -> index fn, N)]()
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: // CHECK-LABEL: kgen.func @"invokeTwice,M=10"
// COM: // CHECK-NEXT: kgen.call @"invokeWithN,N=1
// COM: // CHECK-NEXT: kgen.call @"invokeWithN,N=2
// COM: kgen.generator @invokeTwice<M>() {
// COM:   kgen.param.declare.region fn0 = <N>() -> index {
// COM:     %1 = kgen.param.constant = <add(N, M)>
// COM:     kgen.return %1 : index
// COM:   }
// COM:   %0 = kgen.call @invokeWithN<N = 1, fn: <N>() -> index = fn0>() : () -> index
// COM:   kgen.param.declare.region fn1 = <N>() -> index {
// COM:     %1 = kgen.param.constant = <mul(N, M)>
// COM:     kgen.return %1 : index
// COM:   }
// COM:   %1 = kgen.call @invokeWithN<N = 2, fn: <N>() -> index = fn1>() : () -> index
// COM:   kgen.return
// COM: }
// COM:
// COM: // CHECK-LABEL: kgen.func @doIt
// COM: // CHECK-NEXT: kgen.call @"invokeTwice,M=10"
// COM: kgen.generator @doIt() {
// COM:   kgen.call @invokeTwice<M = 10>() : () -> ()
// COM:   kgen.return
// COM: }

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: // This test case caught a tricky bug with name shadowing.
// COM:
// COM: // CHECK-LABEL: kgen.func @"invokeWithN,N=1
// COM: // CHECK-NEXT: constant = <11>
// COM:
// COM: // CHECK-LABEL: kgen.func @"invokeWithN,N=2
// COM: // CHECK-NEXT: constant = <20>
// COM:
// COM: kgen.generator @invokeWithN<N, fn: <N>() -> index>() -> index{
// COM:   %0 = kgen.call_param[() -> index: bind_signature(:<N>() -> index fn, N)]()
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: kgen.generator @aliasN<N>() {
// COM:   kgen.param.declare M = <N>
// COM:   kgen.param.declare.region fn0 = <N>() -> index {
// COM:     %1 = kgen.param.constant = <add(N, M)>
// COM:     kgen.return %1 : index
// COM:   }
// COM:   %0 = kgen.call @invokeWithN<N = 1, fn: <N>() -> index = fn0>() : () -> index
// COM:   kgen.param.declare.region fn1 = <N>() -> index {
// COM:     %1 = kgen.param.constant = <mul(N, M)>
// COM:     kgen.return %1 : index
// COM:   }
// COM:   %1 = kgen.call @invokeWithN<N = 2, fn: <N>() -> index = fn1>() : () -> index
// COM:   kgen.return
// COM: }
// COM:
// COM: kgen.generator @doIt() {
// COM:   kgen.call @aliasN<N = 10>() : () -> ()
// COM:   kgen.return
// COM: }

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: // CHECK-LABEL: kgen.func @"nestMe,fn=tripleNested,A=1_region_concrete_region_concrete_region"
// COM: // CHECK-NEXT: kgen.param.constant = <6>
// COM: kgen.generator @nestMe<fn: () -> index>() -> index {
// COM:   %0 = kgen.call_param[() -> index: fn]()
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: kgen.generator @tripleNested<A>() -> index{
// COM:   kgen.param.declare.region fn = () -> index {
// COM:     kgen.param.declare B = <2>
// COM:     kgen.param.declare.region fn = () -> index {
// COM:       kgen.param.declare C = <3>
// COM:       kgen.param.declare.region fn = () -> index {
// COM:         %3 = kgen.param.constant = <add(A, B, C)>
// COM:         kgen.return %3 : index
// COM:       }
// COM:       %2 = kgen.call @nestMe<fn: () -> index = fn>() : () -> index
// COM:       kgen.return %2 : index
// COM:     }
// COM:     %1 = kgen.call @nestMe<fn: () -> index = fn>() : () -> index
// COM:     kgen.return %1 : index
// COM:   }
// COM:   %0 = kgen.call @nestMe<fn: () -> index = fn>() : () -> index
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: kgen.generator @doIt() {
// COM:   %0 = kgen.call @tripleNested<A=1>() : () -> index
// COM:   kgen.return
// COM: }

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: // CHECK-LABEL: kgen.func @"nestMe,N=6,fn=tripleNested,A=1_region,N=2_region,N=4_region"
// COM: // CHECK-NEXT: kgen.param.constant = <21>
// COM:
// COM: kgen.generator @nestMe<N, fn: <N>() -> index>() -> index {
// COM:   %0 = kgen.call_param[() -> index: bind_signature(:<N>() -> index fn, N)]()
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: kgen.generator @tripleNested<A>() -> index{
// COM:   kgen.param.declare.region fn = <N>() -> index {
// COM:     kgen.param.declare B = <N>
// COM:     kgen.param.declare C = <3>
// COM:     kgen.param.declare.region fn = <N>() -> index {
// COM:       kgen.param.declare D = <N>
// COM:       kgen.param.declare E = <5>
// COM:       kgen.param.declare.region fn = <N>() -> index {
// COM:       kgen.param.declare F = <N>
// COM:         %3 = kgen.param.constant = <add(A, B, C, D, E, F)>
// COM:         kgen.return %3 : index
// COM:       }
// COM:       %2 = kgen.call @nestMe<N = 6, fn: <N>() -> index = fn>() : () -> index
// COM:       kgen.return %2 : index
// COM:     }
// COM:     %1 = kgen.call @nestMe<N = 4, fn: <N>() -> index = fn>() : () -> index
// COM:     kgen.return %1 : index
// COM:   }
// COM:   %0 = kgen.call @nestMe<N = 2, fn: <N>() -> index = fn>() : () -> index
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: kgen.generator @doIt() {
// COM:   %0 = kgen.call @tripleNested<A=1>() : () -> index
// COM:   kgen.return
// COM: }

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: kgen.generator @hasReturn<() -> index>() {
// COM:   kgen.return<2>
// COM: }
// COM:
// COM: // CHECK-LABEL: kgen.func @"placeholder
// COM: kgen.generator @placeholder<fn: () -> index>() -> index {
// COM:   // CHECK-NEXT: kgen.param.constant = <2>
// COM:   %0 = kgen.call_param[() -> index: fn]()
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: kgen.generator @returnValueOverwritesParameter<SomeParam: dtype>() {
// COM:   kgen.param.declare.region fn = () -> index {
// COM:     %0 = kgen.param.constant = <SomeParam>
// COM:     kgen.call @hasReturn<() -> SomeParam>() : () -> ()
// COM:     kgen.return %0 : index
// COM:   }
// COM:   %0 = kgen.call @placeholder<fn: () -> index = fn>() : () -> index
// COM:   kgen.return
// COM: }
// COM:
// COM: kgen.generator @top() {
// COM:   kgen.call @returnValueOverwritesParameter<SomeParam: dtype = f32>() : () -> ()
// COM:   kgen.return
// COM: }

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: // CHECK-LABEL: kgen.func @passNonIsolatedRegion
// COM: kgen.generator @passNonIsolatedRegion() {
// COM:   // CHECK-NEXT: kgen.call @"callARegion,fn=passNonIsolatedRegion
// COM:   kgen.param.declare.region fn = () {
// COM:     kgen.return
// COM:   }
// COM:   kgen.call @callARegion<fn: () -> () = fn>() : () -> ()
// COM:   kgen.return
// COM: }
// COM:
// COM: kgen.generator @callARegion<fn: () -> ()>() {
// COM:   kgen.call @noReallyCallIt<fn: () -> () = fn>() : () -> ()
// COM:   kgen.return
// COM: }
// COM:
// COM: kgen.generator @noReallyCallIt<fn: () -> ()>() {
// COM:   kgen.return
// COM: }

// -----

// CHECK-LABEL: kgen.func @"takeStringParameter,SomeString=foo"
kgen.generator @takeStringParameter<SomeString: string>()
    constraints <[eq(:string SomeString, "foo"), "I want foo"]> {
  kgen.return
}

// CHECK-LABEL: kgen.func @giveString
kgen.generator @giveString() {
  // CHECK-NEXT: kgen.call @"takeStringParameter,SomeString=foo"
  kgen.call @takeStringParameter<SomeString: string = "foo">() : () -> ()
  kgen.return
}


// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: // CHECK-LABEL: @"pasteTwice
// COM: kgen.generator @pasteTwice<fn: () -> ()>() {
// COM:   // CHECK-NEXT: kgen.call @makeResult
// COM:   // CHECK-NEXT: kgen.call @makeResult
// COM:   kgen.call_param[() -> (): fn]()
// COM:   kgen.call_param[() -> (): fn]()
// COM:   kgen.return
// COM: }
// COM:
// COM: kgen.generator @bindResult() {
// COM:   kgen.param.declare.region fn = () {
// COM:     kgen.call @makeResult<() -> ResultParam>() : () -> ()
// COM:     kgen.return
// COM:   }
// COM:   kgen.call @pasteTwice<fn: () -> () = fn>() : () -> ()
// COM:   kgen.return
// COM: }
// COM:
// COM: kgen.generator @makeResult<() -> index>() {
// COM:   kgen.return<0>
// COM: }

// -----

// CHECK-LABEL: @"makeListConst,A=1"
kgen.generator @makeListConst<A>() {
  // CHECK-NEXT: kgen.param.constant: list<index[2]> = <[1, 1]>
  %0 = kgen.param.constant: list<index[2]> = <[A, A]>
  kgen.return
}

kgen.generator @doIt() {
  kgen.call @makeListConst<A = 1>() : () -> ()
  kgen.return
}

// CHECK-LABEL: @"variableList,N=2,Ts=[1,2]"
kgen.generator @variableList<N, Ts: list<i32[N]>>() {
  // CHECK-NEXT: kgen.param.constant: list<i32[2]> = <[1, 2]>
  %0 = kgen.param.constant: list<i32[N]> = <Ts>
  kgen.return
}

kgen.generator @passTypeList() {
  kgen.call @variableList<N = 2, Ts: list<i32[2]> = [1, 2]>() : () -> ()
  kgen.return
}

kgen.generator @type_of_unknown<T: type, value: !kgen.paramref<T> -> is_unknown: i1>() {
  kgen.return<:i1 eq(:!kgen.paramref<T> value, ?)>
}

// CHECK-LABEL: @check
kgen.generator @check() {
  kgen.call @type_of_unknown<T: type = i32, value: i32 = 1 -> result = is_unknown: i1>() : () -> ()
  // CHECK: = <0>
  %0 = kgen.param.constant: i1 = <result>
  kgen.return
}

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: // CHECK-LABEL: kgen.func @"call_it_nested
// COM: kgen.generator @call_it_nested<fn: <fn: (index) -> index>(index) -> index>(%arg0: index) -> index {
// COM:   // CHECK-NEXT: %idx1 = index.constant 1
// COM:   // CHECK-NEXT: %0 = index.add %arg0, %idx1
// COM:   kgen.param.declare.region fn0 = (%arg1: index) -> index {
// COM:     %0 = index.add %arg0, %arg1
// COM:     kgen.return %0 : index
// COM:   }
// COM:   %1 = kgen.call_param[(index) -> index: bind_signature(:<fn: (index) -> index>(index) -> index fn, fn0)](%arg0)
// COM:   // CHECK-NEXT: kgen.return %0
// COM:   kgen.return %1 : index
// COM: }
// COM:
// COM: // CHECK-LABEL: @call_nested
// COM: kgen.generator @call_nested(%arg0: index) -> index {
// COM:   // CHECK-NEXT: %0 = kgen.call @"call_it_nested{{.*}}(%arg0)
// COM:   kgen.param.declare.region fn = <fn: (index) -> index>(%arg1: index) -> index {
// COM:     %1 = index.constant 1
// COM:     %2 = kgen.call_param[(index) -> index: fn](%1)
// COM:     kgen.return %2 : index
// COM:   }
// COM:   %0 = kgen.call @call_it_nested<fn: <fn: (index) -> index>(index) -> index = fn>(%arg0) : (index) -> index
// COM:   // CHECK-NEXT: return %0
// COM:   kgen.return %0 : index
// COM: }

// -----

//===----------------------------------------------------------------------===//
// Recursion Test
//===----------------------------------------------------------------------===//
//
// This shows that we properly support recursive expansion.
//

kgen.generator.interface @genItf3<x>()

kgen.generator @genItf3_impl0<x>()
  constraints <[eq(x, 0), "x must be zero"]> implements @genItf3 {
  "impl.0"() {attr=#kgen.param.decl.ref<"x"> : index}: () -> ()
  kgen.return
}

kgen.generator @genItf3_impl1<x>()
  constraints <[ne(x, 0), "x must not be zero"]> implements @genItf3 {
  "impl.1"() {attr=#kgen.param.decl.ref<"x"> : index} : () -> ()
  kgen.call @genItf3<x = sub(x, 1)>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @"genItf3_impl0,x=0"()
// CHECK-NEXT:   "impl.0"() {attr = 0 : index}

// CHECK-LABEL: kgen.func @"genItf3_impl1,x=4"()
// CHECK-NEXT:   "impl.1"() {attr = 4 : index}
// CHECK-NEXT:   kgen.call @"genItf3_impl1,x=3"()

// CHECK-LABEL: kgen.func @"genItf3_impl1,x=3"()
// CHECK-NEXT:   "impl.1"() {attr = 3 : index}
// CHECK-NEXT:   kgen.call @"genItf3_impl1,x=2"()

// CHECK-LABEL: kgen.func @"genItf3_impl1,x=2"()
// CHECK-NEXT:   "impl.1"() {attr = 2 : index}
// CHECK-NEXT:   kgen.call @"genItf3_impl1,x=1"()

// CHECK-LABEL: kgen.func @"genItf3_impl1,x=1"()
// CHECK-NEXT:   "impl.1"() {attr = 1 : index}
// CHECK-NEXT:   kgen.call @"genItf3_impl0,x=0"()

// CHECK-LABEL:   kgen.func @use_Itf3() {
// CHECK-NEXT:      kgen.call @"genItf3_impl1,x=4"() : () -> ()
// CHECK-NEXT:      kgen.call @"genItf3_impl1,x=2"() : () -> ()
// CHECK-NEXT:      kgen.return
kgen.generator @use_Itf3() {
  kgen.call @genItf3<x = 4>() : () -> ()
  kgen.call @genItf3<x = 2>() : () -> ()
  kgen.return
}

// -----

kgen.generator @fma(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = index.mul %arg1, %arg2
  %1 = index.add %0, %arg0
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func @constexpr_fma
kgen.generator @constexpr_fma() -> index {
  // CHECK-NEXT: kgen.param.constant = <7>
  %0 = kgen.param.constant = <apply(:(index, index, index) -> index @fma, 1, 2, 3)>
  kgen.return %0 : index
}

// -----

kgen.generator @alloc_load_store(%arg0: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %idx3 = index.constant 3

  %p0 = pop.stack_allocation 4 x index
  pop.store %idx0, %p0 : !pop.pointer<index>
  %p1 = pop.offset %p0[%idx1] : !pop.pointer<index>
  pop.store %idx1, %p1 : !pop.pointer<index>
  %p2 = pop.offset %p0[%idx2] : !pop.pointer<index>
  pop.store %idx2, %p2 : !pop.pointer<index>
  %p3 = pop.offset %p1[%idx2] : !pop.pointer<index>
  pop.store %idx3, %p3 : !pop.pointer<index>

  %ptr = pop.offset %p0[%arg0] : !pop.pointer<index>
  %result = pop.load %ptr : !pop.pointer<index>
  kgen.return %result : index
}

// CHECK-LABEL: kgen.func @constexpr_load_store
kgen.generator @constexpr_load_store() {
  // CHECK-NEXT: = <0>
  %0 = kgen.param.constant = <apply(:(index) -> index @alloc_load_store, 0)>
  // CHECK-NEXT: = <1>
  %1 = kgen.param.constant = <apply(:(index) -> index @alloc_load_store, 1)>
  // CHECK-NEXT: = <2>
  %2 = kgen.param.constant = <apply(:(index) -> index @alloc_load_store, 2)>
  // CHECK-NEXT: = <3>
  %3 = kgen.param.constant = <apply(:(index) -> index @alloc_load_store, 3)>
  kgen.return
}

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: // CHECK-LABEL: kgen.func @bind_signature_region
// COM: kgen.generator @bind_signature_region() -> index {
// COM:   // CHECK-NEXT: %0 = kgen.param.constant = <1>
// COM:   kgen.param.declare.region Fn = <A>() -> index {
// COM:     %0 = kgen.param.constant = <A>
// COM:     kgen.return %0 : index
// COM:   }
// COM:   kgen.param.declare BoundFn: () -> index = <bind_signature(:<A>() -> index Fn, 1)>
// COM:   %0 = kgen.call_param[() -> index: BoundFn]()
// COM:   // CHECK-NEXT: kgen.return %0
// COM:   kgen.return %0 : index
// COM: }

// -----
// COM: This is commented out because the elaborator doesn't currently support regions, but it will in the future.

// COM: // CHECK-LABEL: kgen.func @partial_bind_signature_region
// COM: kgen.generator @partial_bind_signature_region() -> index {
// COM:   // CHECK-NEXT: %0 = kgen.param.constant = <1>
// COM:   kgen.param.declare.region Fn = <A, B>() -> index {
// COM:     %0 = kgen.param.constant = <sub(A, B)>
// COM:     kgen.return %0 : index
// COM:   }
// COM:   kgen.param.declare BoundFn: <A>() -> index = <bind_signature(:<A, B>() -> index Fn, #kgen.unbound, 1)>
// COM:   %0 = kgen.call_param[() -> index: bind_signature(:<A>() -> index BoundFn, 2)]()
// COM:   // CHECK-NEXT: kgen.return %0
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: // CHECK-LABEL: kgen.func @partial_bind_signature_region_2
// COM: kgen.generator @partial_bind_signature_region_2() -> index {
// COM:   // CHECK-NEXT: %0 = kgen.param.constant = <3>
// COM:   kgen.param.declare.region Fn = <A, B>() -> index {
// COM:     %0 = kgen.param.constant = <add(A, B)>
// COM:     kgen.return %0 : index
// COM:   }
// COM:   kgen.param.declare BoundFn: <B>() -> index = <bind_signature(:<A, B>() -> index Fn, 1, #kgen.unbound)>
// COM:   %0 = kgen.call_param[() -> index: bind_signature(:<B>() -> index BoundFn, 2)]()
// COM:   // CHECK-NEXT: kgen.return %0
// COM:   kgen.return %0 : index
// COM: }
// COM:
// COM: // CHECK-LABEL: kgen.func @partial_bind_signature_region_3
// COM: kgen.generator @partial_bind_signature_region_3() -> index {
// COM:   // CHECK-NEXT: %0 = kgen.param.constant = <4>
// COM:   kgen.param.declare.region Fn = <A, B, C>() -> index {
// COM:     %0 = kgen.param.constant = <add(sub(B, A), C)>
// COM:     kgen.return %0 : index
// COM:   }
// COM:   kgen.param.declare BoundFn: <B, C>() -> index = <bind_signature(:<A, B, C>() -> index Fn, 1, #kgen.unbound, #kgen.unbound)>
// COM:   kgen.param.declare BoundFn2: <B>() -> index = <bind_signature(:<B, C>() -> index BoundFn, #kgen.unbound, 3)>
// COM:   %0 = kgen.call_param[() -> index: bind_signature(:<B>() -> index BoundFn2, 2)]()
// COM:   // CHECK-NEXT: kgen.return %0
// COM:   kgen.return %0 : index
// COM: }

// CHECK-LABEL: kgen.func @"param_add,A=1,B=2"
kgen.generator @param_add<A, B>() -> index {
  // CHECK-NEXT: %0 = kgen.param.constant = <3>
  %0 = kgen.param.constant = <add(A, B)>
  // CHECK-NEXT: kgen.return %0
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @partial_bind_signature_region_4
kgen.generator @partial_bind_signature_region_4() -> index {
  kgen.param.declare BoundFn: <B>() -> index = <bind_signature(:<A, B>() -> index @param_add, 1, #kgen.unbound)>
  // CHECK-NEXT: %0 = kgen.call @"param_add,A=1,B=2"() : () -> index
  %0 = kgen.call_param[() -> index: bind_signature(:<B>() -> index BoundFn, 2)]()
  // CHECK-NEXT: kgen.return %0
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @"param_add3,A=1,B=2,C=3"
kgen.generator @param_add3<A, B, C>() -> index {
  // CHECK-NEXT: %0 = kgen.param.constant = <4>
  %0 = kgen.param.constant = <add(sub(B, A), C)>
  // CHECK-NEXT: kgen.return %0
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @partial_bind_signature_region_5
kgen.generator @partial_bind_signature_region_5() -> index {
  kgen.param.declare BoundFn: <B, C>() -> index = <bind_signature(:<A, B, C>() -> index @param_add3, 1, #kgen.unbound, #kgen.unbound)>
  kgen.param.declare BoundFn2: <B>() -> index = <bind_signature(:<B, C>() -> index BoundFn, #kgen.unbound, 3)>
  // CHECK-NEXT: %0 = kgen.call @"param_add3,A=1,B=2,C=3"() : () -> index
  %0 = kgen.call_param[() -> index: bind_signature(:<B>() -> index BoundFn2, 2)]()
  // CHECK-NEXT: kgen.return %0
  kgen.return %0 : index
}

// -----

kgen.generator @return_it<A>() -> index {
  %0 = kgen.param.constant = <A>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:() -> index bind_signature(:<A>() -> index @return_it, 1))>
  // CHECK-NEXT: <2>
  kgen.param.constant = <apply(:() -> index bind_signature(:<A>() -> index @return_it, 2))>
  // CHECK-NEXT: <3>
  kgen.param.constant = <apply(:() -> index bind_signature(:<A>() -> index @return_it,
    apply(:() -> index bind_signature(:<A>() -> index @return_it, 3))))>
  kgen.return
}

// -----

kgen.generator @callee(%arg0: index) -> index {
  %0 = index.add %arg0, %arg0
  kgen.return %0 : index
}

kgen.generator @func(%arg0: index) -> index {
  %0 = kgen.call @callee(%arg0) : (index) -> index
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() -> index {
  // CHECK-NEXT: <14>
  %0 = kgen.param.constant = <apply(:(index) -> index @func, 7)>
  kgen.return %0 : index
}

// -----

kgen.generator @sum(%from: index, %to: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %result = hlcf.loop (%acc = %idx0 : index, %i = %from : index) -> index {
    %cond = index.cmp sle(%i, %to)
    hlcf.if %cond {
      hlcf.yield
    } else {
      hlcf.break %acc : index
    }
    %nextI = index.add %idx1, %i
    %nextAcc = index.add %acc, %i
    hlcf.continue %nextAcc, %nextI : index, index
  }
  kgen.return %result : index
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() {
  // CHECK-NEXT: <55>
  kgen.param.constant = <apply(:(index, index) -> index @sum, 0, 10)>
  kgen.return
}

// -----

kgen.generator @early_return(%cond: i1) -> index {
  %idx0 = index.constant 0
  %result = hlcf.if %cond -> index {
    hlcf.yield %idx0 : index
  } else {
    %idx1 = index.constant 1
    hlcf.return %idx1 : index
  }
  kgen.return %result : index
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:(i1) -> index @early_return, 0)>
  // CHECK-NEXT: <0>
  kgen.param.constant = <apply(:(i1) -> index @early_return, 1)>
  kgen.return
}

kgen.generator @rebind_value<dtype: dtype>(%a: !pop.scalar<ui8>) -> !pop.scalar<dtype> {
  %result = kgen.rebind %a : !pop.scalar<ui8> to !pop.scalar<dtype>
  kgen.return %result : !pop.scalar<dtype>
}

// CHECK-LABEL: kgen.func @rebind_it
kgen.generator @rebind_it() {
  // CHECK-NEXT: constant: scalar<ui8> = <4>
  kgen.param.declare Fn: (!pop.scalar<ui8>) -> !pop.scalar<ui8> =
    <bind_signature(:<dtype: dtype>(!pop.scalar<ui8>) -> !pop.scalar<dtype> @rebind_value, ui8)>
  kgen.param.constant: scalar<ui8> = <apply(:(!pop.scalar<ui8>) -> !pop.scalar<ui8> Fn, <4>)>
  kgen.return
}

kgen.generator.interface @interpretedItf<I>() -> index

// CHECK-LABEL: kgen.func @"interpretedItf.1,I=0"
kgen.generator @interpretedItf.1<I>() -> index constraints <[eq(I, 0), "0"]> implements @interpretedItf {
  // CHECK-NEXT: index.constant 0
  %0 = index.constant 0
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @"interpretedItf.2,I=1"
kgen.generator @interpretedItf.2<I>() -> index constraints <[eq(I, 1), "1"]> implements @interpretedItf {
  // CHECK-NEXT: index.constant 1
  %0 = index.constant 1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @interpretInterfaceCall
kgen.generator @interpretInterfaceCall() {
  // CHECK-NEXT: <0>
  kgen.param.constant = <apply(:() -> index bind_signature(:<I>() -> index @interpretedItf, 0))>
  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:() -> index bind_signature(:<I>() -> index @interpretedItf, 1))>
  kgen.return
}

// CHECK-NOT: @itfIsNeverCalled
// CHECK-NOT: @alwaysBadImpl
kgen.generator.interface @itfIsNeverCalled()
kgen.generator @alwaysBadImpl()
    constraints <[0, "always bad"]>
    implements @itfIsNeverCalled {
  kgen.return
}

// -----

kgen.generator @result<() -> x>() {
  kgen.return<3>
}

// CHECK-LABEL @"add,x=3,y=1"
// CHECK-LABEL @"add,x=3,y=2"
kgen.generator @add<x, y>() -> index {
  %0 = kgen.param.constant = <add(x, y)>
  kgen.return %0 : index
}

kgen.generator @multiVersion() -> index {
  kgen.call @result<() -> x = x>() : () -> ()
  kgen.param.search y = <1, 2>
  %0 = kgen.call @add<x = x, y = y>() : () -> index
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: @"g1,size=5"
// CHECK-LABEL: @"g1,size=3"
kgen.generator @g1<size>() -> index {
  %0 = kgen.param.constant = <size>
  kgen.return %0 : index
}

// CHECK-LABEL: @"g2,size=3,width=5"
kgen.generator @g2<size, width>() -> index {
  // CHECK-NEXT: call @"g1,size=5"
  %0 = kgen.call @g1<size = width>() : () -> index
  // CHECK-NEXT: call @"g1,size=3"
  %1 = kgen.call @g1<size = size>() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: @root
kgen.generator @root() {
  kgen.param.declare q = <3>
  kgen.param.declare w = <5>
  // CHECK-NEXT: kgen.call @"g2,size=3,width=5"
  %0 = kgen.call @g2<size = q, width = w>() : () -> index
  kgen.return
}

// -----

// CHECK-LABEL: @top
kgen.generator @top() {
  // CHECK-NEXT: call @top_impl
  kgen.call @top_itf() : () -> ()
  kgen.return
}

// COM: This interface has an evaluator, so we will be picking the best candidate
// COM: for it.
kgen.generator.interface @top_itf() -> ()
evaluator (!pop.pointer<() -> ()>, index) -> index = @eval


// COM: An evaluator: simply return non-zeroth option. Picking 0th candidate
// COM: works fine.
// CHECK-LABEL: @eval
kgen.generator @eval(%funcs: !pop.pointer<() -> ()>, %size: index) -> index {
  // CHECK-NEXT: index.constant 1
  %res = index.constant 1
  // CHECK-NEXT: return %idx1
  kgen.return %res : index
}

// COM: A single implementation of top_itf where we just call another interface.
// CHECK-LABEL: @top_impl()
kgen.generator @top_impl() -> () implements @top_itf {
  // CHECK-NEXT: call @itf_impl_1
  kgen.call @itf(): ()->()
  kgen.return
}

// COM: This interface has just two implementations but we choose the second
// COM: through the evaluator.
kgen.generator.interface @itf() -> ()

kgen.generator @itf_impl_0() -> () implements @itf {
  kgen.return
}

kgen.generator @itf_impl_1() -> () implements @itf {
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
  %0 = kgen.param.if <lt(cond_var, 10) -> next> -> index {
    %1 = "should.not.appear"() : () -> index
    kgen.param.declare next_lt = <add(cond_var, 10)>
    kgen.param.yield<next_lt> %1 : index
  } else {
    %3 = "should.appear"() : () -> index
    kgen.param.declare next_gt = <add(cond_var, 20)>
    kgen.param.yield<next_gt> %3 : index
  }
  // CHECK-NEXT: param.constant = <52>
  %4 = kgen.param.constant = <next>

  kgen.return
}

// CHECK-LABEL: @nestedConstexprIf()
kgen.generator @nestedConstexprIf() {
  kgen.param.declare cond_var = <32>

  // CHECK-NEXT: "should.appear"
  %0 = kgen.param.if <lt(cond_var, 10) -> next> -> index {
    %1 = "should.not.appear"() : () -> index
    kgen.param.declare next_lt = <add(cond_var, 10)>
    kgen.param.yield<next_lt> %1 : index
  } else {
    %3 = kgen.param.if <gt(cond_var, 30) -> next_gt> -> index {
      %4 = "should.appear"() : () -> index
      kgen.param.declare next_gt_gt = <add(cond_var, 20)>
      kgen.param.yield<next_gt_gt> %4 : index
    } else {
      %4 = "should.not.appear"() : () -> index
      kgen.param.declare next_gt_lt = <add(cond_var, 1)>
      kgen.param.yield<next_gt_lt> %4 : index
    }
    kgen.param.yield<next_gt> %3 : index
  }
  // CHECK-NEXT: param.constant = <52>
  %4 = kgen.param.constant = <next>

  kgen.return
}

// -----

kgen.generator @someFunc<x>() {
  kgen.return
}

// CHECK-LABEL: @constexprIfWithSearch_concrete_2()
// CHECK-NEXT:   "should.appear"
// CHECK-NEXT:   "someFunc,x=3"
// CHECK-NEXT:   param.constant = <42>

// CHECK-LABEL: @constexprIfWithSearch_concrete_1()
// CHECK-NEXT:   "should.appear"
// CHECK-NEXT:   "someFunc,x=2"
// CHECK-NEXT:   param.constant = <42>

// CHECK-LABEL: @constexprIfWithSearch()
// CHECK-NEXT:   "should.appear"
// CHECK-NEXT:   "someFunc,x=1"
// CHECK-NEXT:   param.constant = <42>

kgen.generator @constexprIfWithSearch() {
  kgen.param.declare cond_var = <32>
  kgen.param.search inParam = <1, 2, 3>

  %0 = kgen.param.if <gt(cond_var, 10) -> next> -> index {
    %1 = "should.appear"() : () -> index
    kgen.call @someFunc<x = inParam>() : () -> ()
    kgen.param.declare next_lt = <add(cond_var, 10)>
    kgen.param.yield<next_lt> %1 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.call @someFunc<x = inParam>() : () -> ()
    kgen.param.declare next_gt = <add(cond_var, 20)>
    kgen.param.yield<next_gt> %3 : index
  }
  %4 = kgen.param.constant = <next>

  kgen.return
}

// -----

kgen.generator @someFunc<x -> y>() {
  kgen.return<and(x, 2)>
}

// CHECK-LABEL: @constexprIfWithReturnedCondition_concrete_2()
// CHECK-NEXT:   "someFunc,x=3"
// COM: This should be 12 because we have (3 & 2) + 10 == 12
// CHECK-NEXT:   param.constant = <12>

// CHECK-LABEL: @constexprIfWithReturnedCondition_concrete_1()
// CHECK-NEXT:   "someFunc,x=2"
// COM: This should be 12 because we have (2 & 2) + 10 == 12
// CHECK-NEXT:   param.constant = <12>

// CHECK-LABEL: @constexprIfWithReturnedCondition()
// CHECK-NEXT:   "someFunc,x=1"
// COM: This should be 20 because we have (1 & 2) + 20 == 20
// CHECK-NEXT:   param.constant = <20>

kgen.generator @constexprIfWithReturnedCondition() {
  kgen.param.search inParam = <1, 2, 3>

  kgen.param.if <eq(cond_var, 2) -> next> {
    kgen.param.declare next_lt = <add(cond_var, 10)>
    kgen.param.yield<next_lt>
  } else {
    kgen.param.declare next_gt = <add(cond_var, 20)>
    kgen.param.yield<next_gt>
  }

  kgen.call @someFunc<x = inParam -> cond_var = y>() : () -> ()

  %4 = kgen.param.constant = <next>

  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @substitute_current_target
kgen.generator @substitute_current_target() {
  // CHECK-NEXT: constant: target = <#kgen.target<triple = {{.*}}>>
  kgen.param.constant: target = <current_target()>
  kgen.return
}
