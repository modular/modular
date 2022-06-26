// RUN: kgen-generate %s -library=%S/library.mlir | FileCheck %s

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


// TODO: enable this as a kernel test eventually.
kgen.generator @test1(%arg0: si32, %arg1: si32) -> (si32, si32, si32) {
  // Can invoke the generator directly.
  %0 = kgen.call @trivial_generator(%arg0) : (si32) -> si32

  // Can invoke the interface, binding a parameter.
  %1 = kgen.call @unary_add<size = 42>(%arg0) : (si32) -> si32

  // Can invoke parameterized generators directly as well.
  %2 = kgen.call @unary_add_impl1<size = 12>(%arg0) : (si32) -> si32

  kgen.return %0, %1, %2 : si32, si32, si32
}
