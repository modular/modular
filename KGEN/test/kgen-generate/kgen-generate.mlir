// RUN: kgen-generate %s -library=%S/library.mlir -verify-diagnostics | FileCheck %s

// CHECK-NOT: kgen.generator @trivial_generator
kgen.generator @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-NOT: kgen.generator.interface @unary_add
kgen.generator.interface @unary_add<size>(si32) -> si32

kgen.generator @unary_add_impl1<size>(%arg0: si32) -> si32
  implements @unary_add {

  // Silly op so we know when something used this.
  "unary_add_impl1"() : () -> ()

  // TODO: Do something with <size>
  kgen.return %arg0 : si32
}

// CHECK: kgen.kernel @test0() -> index {
// CHECK-NEXT: %0 = kgen.param.value = <1>
// CHECK-NEXT:  kgen.return %0 : index
// CHECK-NEXT: }
kgen.kernel @test0() -> index {
  %0 = kgen.param.value = <1>
  kgen.return %0 : index
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

// TODO: enable this as a kernel test eventually.
kgen.generator @test_xx(%arg0: si32, %arg1: si32) -> (si32, si32, si32) {
  // Can invoke the generator directly.
  %0 = kgen.call @trivial_generator(%arg0) : (si32) -> si32

  // Can invoke the interface, binding a parameter.
  %1 = kgen.call @unary_add<size = 42>(%arg0) : (si32) -> si32

  // Can invoke parameterized generators directly as well.
  %2 = kgen.call @unary_add_impl1<size = 12>(%arg0) : (si32) -> si32

  kgen.return %0, %1, %2 : si32, si32, si32
}

kgen.generator.interface @take_and_return<p1 -> r1>()

kgen.generator @parameter_call_use_chain() {
  // Uses r2 and defines r1
  kgen.call @take_and_return<p1 = r2 -> r1>() { someAttr = 1 } : () -> ()

  // Uses 42 and defines r2
  kgen.call @take_and_return<p1 = 42 -> r2>() { someAttr = 2 }: () -> ()

  // Uses r1 and defines r3
  kgen.call @take_and_return<p1 = r1 -> r3>() { someAttr = 3 }: () -> ()

  kgen.return
}
