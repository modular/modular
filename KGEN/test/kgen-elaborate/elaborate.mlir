// RUN: kgen-elaborate %s -library=%S/library-test.mlir | FileCheck %s

// This is left untouched.
// CHECK-LABEL: kgen.kernel @test0<() -> outParam>() -> index {
// CHECK-NEXT: %0 = kgen.param.value = <1>
// CHECK-NEXT:  kgen.return <outParam = 123456> %0 : index
// CHECK-NEXT: }
kgen.kernel @test0<() -> outParam>() -> index {
  %0 = kgen.param.value = <1>
  kgen.return <outParam = 123456> %0 : index
}

// CHECK-LABEL: kgen.kernel @parameter_use_chain()
kgen.kernel @parameter_use_chain() {
  // Uses r2 and defines r1
  kgen.param.bind r1 = <add(r2, 1)>
  // CHECK-NEXT: %0 = kgen.param.value = <3>
  %0 = kgen.param.value = <r1>

  // Uses 42 and defines r2
  kgen.param.bind r2 = <2>
  // CHECK-NEXT: %1 = kgen.param.value = <2>
  %1 = kgen.param.value = <r2>

  // Uses r1/r2 and defines r3
  kgen.param.bind r3 = <mul(shl(r1, r2), 3)>
  // CHECK-NEXT: %2 = kgen.param.value = <36>
  %2 = kgen.param.value = <r3>

  // Defines a dtype value and uses it.
  kgen.param.bind type1 : !kgen.dtype = <f32>
  // CHECK-NEXT: %3 = kgen.param.value : dtype = <f32>
  %3 = kgen.param.value : !kgen.dtype = <type1>

  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-NOT: kgen.generator @trivial_generator
// This gets "specialized" into a kernel.
kgen.generator @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}
// CHECK-LABEL: kgen.kernel @trivial_generator_kernel(%arg0: si32) -> si32 {
// CHECK-NEXT:    kgen.return %arg0 : si32
// CHECK-NEXT: }

kgen.generator @genA<size, type: dtype, val: f32 -> out>(%arg0: si32) -> si32 {

  %0 = kgen.param.value = <add(size, 4)>
  %1 = kgen.param.value : dtype = <type>
  %2 = kgen.param.value : f32 = <val>

  // Silly op so we know when something used this.
  "genA op"() { value = #kgen.param.decl.ref<"size"> : index} : () -> !meta.scalar<type>

  kgen.return<out = mul(size, 2)> %arg0 : si32
}
// CHECK-LABEL: kgen.kernel @"genA,size=42,type=f32,val=2"<() -> out>(%arg0: si32) -> si32 {
// CHECK-NEXT:   %0 = kgen.param.value  = <46>
// CHECK-NEXT:   %1 = kgen.param.value : dtype = <f32>
// CHECK-NEXT:   %2 = kgen.param.value : f32 = <2.000000e+00>
// CHECK-NEXT:   %3 = "genA op"() {value = 42 : index} : () -> !meta.scalar<f32>
// CHECK-NEXT:   kgen.return <out = 84> %arg0 : si32
// CHECK-NEXT: }

// CHECK-LABEL: kgen.kernel @"genA,size=19,type=si8,val=1.5"<() -> out>(%arg0: si32) -> si32 {
// CHECK-NEXT:    %0 = kgen.param.value  = <23>
// CHECK-NEXT:    %1 = kgen.param.value : dtype = <si8>
// CHECK-NEXT:    %2 = kgen.param.value : f32 = <1.500000e+00>
// CHECK-NEXT:    %3 = "genA op"() {value = 19 : index} : () -> !meta.scalar<si8>
// CHECK-NEXT:    kgen.return <out = 38> %arg0 : si32
// CHECK-NEXT:  }

// CHECK-LABEL: kgen.kernel @call_generator_test
kgen.kernel @call_generator_test(%arg0: si32, %arg1: si32)
   -> (si32, si32, si32, index, index) {
  // Can invoke the generator directly.
  %0 = kgen.call @trivial_generator(%arg0) : (si32) -> si32
  // CHECK-NEXT: %0 = kgen.call @trivial_generator_kernel(%arg0)

  // CHECK-NOT: kgen.param.bind
  kgen.param.bind our_size = <42>

  // Can invoke parameterized generators directly.
  %1 = kgen.call @genA<size = our_size, type : dtype = f32, val : f32 = 2.0 -> resultSizeA>(%arg0) : (si32) -> si32
  // CHECK-NEXT: %1 = kgen.call @"genA,size=42,type=f32,val=2"<() -> resultSizeA>(%arg0) : (si32) -> si32

  %2 = kgen.call @genA<size = 19, type : dtype = si8, val : f32 = 1.5 -> resultSizeB>(%arg1) : (si32) -> si32
  // CHECK-NEXT: %2 = kgen.call @"genA,size=19,type=si8,val=1.5"<() -> resultSizeB>(%arg1) : (si32) -> si32

  %3 = kgen.call @genA<size = 19, type : dtype = si8, val : f32 = 1.5 -> resultSizeC>(%arg1) : (si32) -> si32
  // CHECK-NEXT: %3 = kgen.call @"genA,size=19,type=si8,val=1.5"<() -> resultSizeC>(%arg1) : (si32) -> si32


  %4 = kgen.param.value = <resultSizeA>
  // CHECK-NEXT: %4 = kgen.param.value = <84>

  %5 = kgen.param.value = <resultSizeB>
  // CHECK-NEXT: %5 = kgen.param.value = <38>

  %6 = kgen.param.value = <resultSizeC>
  // CHECK-NEXT: %6 = kgen.param.value = <38>

  %7 = kgen.call @test0<() -> kernelResult>() : () -> index
  // CHECK-NEXT: %7 = kgen.call @test0<() -> kernelResult>()

  %8 = kgen.param.value = <kernelResult>
  // CHECK-NEXT: %8 = kgen.param.value = <123456>

  kgen.return %0, %1, %2, %4, %5 : si32, si32, si32, index, index
}

//===----------------------------------------------------------------------===//

// CHECK-NOT: kgen.generator.interface @genItf
kgen.generator.interface @genItf<x -> y>(si32) -> si32

// CHECK-LABEL: kgen.kernel @"genItf_impl1,x=42"<() -> y>(
// CHECK-NEXT:   "genItf_impl1"() {value = 42 : index} : () -> ()
// CHECK-NEXT:   kgen.return <y = 43> %arg0 : si32
// CHECK-NEXT: }
kgen.generator @genItf_impl1<x -> y>(%arg0: si32) -> si32
  implements @genItf {
  "genItf_impl1"() { value = #kgen.param.decl.ref<"x"> : index} : () -> ()
  kgen.return<y = add(x, 1)> %arg0 : si32
}

// CHECK-LABEL: kgen.kernel @"genItf_impl2,x=42"<() -> y>(
// CHECK-NEXT:   "genItf_impl2"() {value = 42 : index} : () -> ()
// CHECK-NEXT:   kgen.return <y = 84> %arg0 : si32
// CHECK-NEXT: }
kgen.generator @genItf_impl2<x -> y>(%arg0: si32) -> si32
  implements @genItf {
  "genItf_impl2"() { value = #kgen.param.decl.ref<"x"> : index} : () -> ()
  kgen.return<y = mul(x, 2)> %arg0 : si32
}

// CHECK-LABEL: kgen.kernel @use_interface(
// CHECK-NEXT: %0 = kgen.call @"genItf_impl1,x=42"<() -> out>(%arg0)
// CHECK-NEXT: %1 = kgen.param.value = <43>

// CHECK-LABEL: kgen.kernel @use_interface_0(%arg0: si32) -> index {
// CHECK-NEXT:    %0 = kgen.call @"genItf_impl2,x=42"<() -> out>(%arg0) : (si32) -> si32
// CHECK-NEXT:     %1 = kgen.param.value = <84>
kgen.kernel @use_interface(%arg0: si32) -> index {
  %0 = kgen.call @genItf<x = 42 -> out>(%arg0) : (si32) -> si32
  %1 = kgen.param.value = <out>
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.kernel @use_kernel_using_interface(%arg0: si32) -> index {
// CHECK-NEXT:   %0 = kgen.call @use_interface(%arg0) : (si32) -> index
// CHECK-NEXT:   kgen.return  %0 : index

// CHECK-LABEL: kgen.kernel @use_kernel_using_interface_1(%arg0: si32) -> index {
// CHECK-NEXT:   %0 = kgen.call @use_interface_0(%arg0) : (si32) -> index
// CHECK-NEXT:   kgen.return  %0 : index
kgen.kernel @use_kernel_using_interface(%arg0: si32) -> index {
  %0 = kgen.call @use_interface(%arg0) : (si32) -> index
  kgen.return %0 : index
}

//===----------------------------------------------------------------------===//

// CHECK-NOT: @genItf2<x>()
kgen.generator.interface @genItf2<x>()

// CHECK-NOT: kgen.kernel @"genItf2_impl0,x=1"() {
// CHECK-LABEL: kgen.kernel @"genItf2_impl0,x=0"() {
// CHECK-NEXT:   "impl0"() : () -> ()
// CHECK-NEXT:   kgen.return
// CHECK-NOT: kgen.kernel @"genItf2_impl0,x=1"() {
kgen.generator @genItf2_impl0<x>()
  constraints <eq(x, 0), "x must be zero"> implements @genItf2 {
  "impl0"() : () -> ()
  kgen.return
}

// CHECK-NOT: kgen.kernel @"genItf2_impl1,x=0"()
// CHECK-LABEL: kgen.kernel @"genItf2_impl1,x=1"() {
// CHECK-NEXT:   "impl1"() : () -> ()
// CHECK-NEXT:   kgen.return
// CHECK-NOT: kgen.kernel @"genItf2_impl1,x=0"()
kgen.generator @genItf2_impl1<x>()
  constraints <eq(x, 1), "x must be 1"> implements @genItf2 {
  "impl1"() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.kernel @use_Itf2zero() {
// CHECK-NEXT:   kgen.call @"genItf2_impl0,x=0"() : () -> ()
// CHECK-NEXT:   kgen.return
kgen.kernel @use_Itf2zero() {
  kgen.call @genItf2<x = 0>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.kernel @use_Itf2one() {
// CHECK-NEXT:   kgen.call @"genItf2_impl1,x=1"() : () -> ()
// CHECK-NEXT:   kgen.return
// CHECK-NEXT: }
kgen.kernel @use_Itf2one() {
  kgen.call @genItf2<x = 1>() : () -> ()
  kgen.return
}

// -----

kgen.generator.interface @genItf3<ty: dtype>()

// This implementation is fine.
// CHECK-LABEL: kgen.kernel @"genItf3_impl0,ty=f32"() {
kgen.generator @genItf3_impl0<ty: dtype>() implements @genItf3 {
  "impl0"() : () -> ()
  kgen.return
}

// This generates a kernel that fails to verify, so it isn't used and must be
// deleted.
// CHECK-NOT: genItf3_impl1
kgen.generator @genItf3_impl1<ty: dtype>() implements @genItf3 {
  %c1 = arith.constant 1.0 : f32
  %0 = meta.cast_from_builtin %c1: f32 to !meta.scalar<ty>
  %1 = meta.cast_to_builtin %0: !meta.scalar<ty> to i8
  kgen.return
}

// This has a single viable implementation.
// CHECK-LABEL: kgen.kernel @use_Itf3() {
// CHECK-NEXT:    kgen.call @"genItf3_impl0,ty=f32"()
kgen.kernel @use_Itf3() {
  kgen.call @genItf3<ty: dtype = f32>() : () -> ()
  kgen.return
}

// -----

// Test that expansions are tracked and each ultimate kernel version only allows
// any particular generator/parameter set pair to expand one direction, reducing
// exponential explosion.

// CHECK-LABEL: kgen.kernel @track_expansions(%arg0: si32)
// CHECK-NEXT: kgen.call @"genItf_impl1,x=42"<() -> out>(%arg0) : (si32) -> si32
// CHECK-NEXT: kgen.call @"genItf_impl1,x=42"<() -> out1>(%arg0) : (si32) -> si32
// CHECK-NEXT: kgen.call @use_interface(%arg0)

// CHECK-NOT: kgen.kernel @track_expansions

// CHECK-LABEL: kgen.kernel @track_expansions_2(%arg0: si32)
// CHECK-NEXT: kgen.call @"genItf_impl2,x=42"
// CHECK-NEXT: kgen.call @"genItf_impl2,x=42"
// CHECK-NEXT: kgen.call @use_interface_0(%arg0)

// CHECK-NOT: kgen.kernel @track_expansions

kgen.kernel @track_expansions(%arg0: si32) {
  // Within any generated kernel genItf should expand the same way.
  %0 = kgen.call @genItf<x = 42 -> out>(%arg0) : (si32) -> si32
  %1 = kgen.call @genItf<x = 42 -> out1>(%arg0) : (si32) -> si32

  // Even if deeply nested within other generator/kernel invocations
  %2 = kgen.call @use_interface(%arg0) : (si32) -> index
  kgen.return
}


// -----

// Test that parameter and result argument types get rewritten and specialized.

// CHECK-LABEL: kgen.kernel @"float_constant_f32,value=1.5,type=f32"() -> !meta.scalar<f32> {
// ...
// CHECK:    %2 = meta.cast_from_builtin %1 : f32 to !meta.scalar<f32>
// CHECK:    kgen.return  %2 : !meta.scalar<f32>

kgen.generator @float_constant_f32<value: f64, type: dtype>() -> !meta.scalar<type>
  constraints <eq(:dtype type, f32), "float please">  {
  %0 = kgen.param.value : f64 = <value>
  %1 = llvm.fptrunc %0 : f64 to f32
  %2 = meta.cast_from_builtin %1: f32 to !meta.scalar<type>
  kgen.return %2 : !meta.scalar<type>
}

// CHECK-LABEL: kgen.kernel @test_f32() -> f32 {
// CHECK:    %0 = kgen.call @"float_constant_f32,value=1.5,type=f32"() : () -> !meta.scalar<f32>
// CHECK:    %1 = meta.cast_to_builtin %0 : !meta.scalar<f32> to f32
kgen.kernel @test_f32() -> f32 {
  kgen.param.bind type : dtype = <f32>
  %1 = kgen.call @float_constant_f32<value: f64 = 1.5, type: dtype = type>() : () -> !meta.scalar<type>
  %2 = meta.cast_to_builtin %1 : !meta.scalar<type> to f32
  kgen.return %2 : f32
}

// -----

// Test that we can do static assertions on computed parameter expressions (e.g.
// those that are the result of a sub-generator invocation.

kgen.generator.interface @getSIMDLength<dt: dtype -> length>()

kgen.generator @getSIMDLengthF32<dt: dtype -> length>() 
     implements @getSIMDLength {
  // This could be implemented as a constraint.
  kgen.param.assert <eq(:dtype dt, f32)>, "this only works for f32"
  // vector length for floats is 4 on our target.
  kgen.return <length = 4>
}

kgen.generator @getSIMDLengthF64<dt: dtype -> length>() 
     implements @getSIMDLength {
  // This could be implemented as a constraint.
  kgen.param.assert <eq(:dtype dt, f64)>, "this only works for f32"
  // vector length for doubles is 2 on our target.
  kgen.return <length = 2>
}

// CHECK-LABEL: kgen.kernel @paramAssertExample()
// CHECK-NEXT:    kgen.call @"getSIMDLengthF32,dt=f32"<() -> flen>() 
// CHECK-NEXT:    kgen.return
kgen.kernel @paramAssertExample() {
  kgen.call @getSIMDLength<dt : dtype = f32 -> flen>() : () -> ()
  
  // Should succeed.
  kgen.param.assert <eq(flen, 4)>, "vector length should be 4 for floats"
  kgen.return
}

