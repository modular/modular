// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor -verify-diagnostics %t.mef | FileCheck %s

// CHECK-LABEL: --- Running 'basic_test'
grt.testcase @basic_test {
  %ch0 = grt.chain.create ()
  %v1 = "test.vec.iota"() { size = 5 : si32 } : () -> !test.vec
  %ch1 = "test.vec.print" (%v1, %ch0) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: vec = 0, 1, 2, 3, 4

  %v2 = "test.vec.constant"() { size = 5 : si32, value = 3 : si32 } : () -> !test.vec
  %ch2 = "test.vec.print" (%v2, %ch1) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: vec = 3, 3, 3, 3, 3

  %v3 = "test.vec.add"(%v1, %v2) : (!test.vec, !test.vec) -> !test.vec
  %ch3 = "test.vec.print" (%v3, %ch2) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: vec = 3, 4, 5, 6, 7

  %v4 = "test.vec.add"(%v3, %v1) : (!test.vec, !test.vec) -> !test.vec
  %ch4 = "test.vec.print" (%v4, %ch3) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: vec = 3, 5, 7, 9, 11

  // This will produce an error due to mismatched lengths.
  %v5 = "test.vec.constant"() { size = 3 : si32, value = 3 : si32 } : () -> !test.vec

  // expected-error @+1 {{vector length mismatch}}
  %v6 = "test.vec.add"(%v5, %v1) : (!test.vec, !test.vec) -> !test.vec
  %ch5 = "test.vec.print" (%v6, %ch4) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: 'basic_test' returned <error: vector length mismatch>

  mef.output %ch5 : !mef.chain
}

// CHECK-LABEL: --- Running 'async_add_test'
grt.testcase @async_add_test {
  %ch0 = grt.chain.create ()
  %v1 = "test.vec.iota"() { size = 5 : si32 } : () -> !test.vec
  %ch1 = "test.vec.print" (%v1, %ch0) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: vec = 0, 1, 2, 3, 4

  %v2 = "test.vec.constant"() { size = 5 : si32, value = 3 : si32 } : () -> !test.vec
  %ch2 = "test.vec.print" (%v2, %ch1) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: vec = 3, 3, 3, 3, 3

  %v3 = "test.vec.add.async"(%v1, %v2) : (!test.vec, !test.vec) -> !test.vec
  %ch3 = "test.vec.print" (%v3, %ch2) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: vec = 3, 4, 5, 6, 7

  %v4 = "test.vec.add.async"(%v3, %v1) : (!test.vec, !test.vec) -> !test.vec
  %ch4 = "test.vec.print" (%v4, %ch3) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: vec = 3, 5, 7, 9, 11

  // This will produce an error due to mismatched lengths.
  %v5 = "test.vec.constant"() { size = 3 : si32, value = 3 : si32 } : () -> !test.vec

  // expected-error @+1 {{vector length mismatch}}
  %v6 = "test.vec.add.async"(%v5, %v1) : (!test.vec, !test.vec) -> !test.vec
  %ch5 = "test.vec.print" (%v6, %ch4) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: 'async_add_test' returned <error: vector length mismatch>

  mef.output %ch5 : !mef.chain
}

// CHECK-LABEL: --- Running 'async_add_test_par_foreach'
grt.testcase @async_add_test_par_foreach {
  %ch0 = grt.chain.create ()
  %v1 = "test.vec.iota"() { size = 5 : si32 } : () -> !test.vec
  %ch1 = "test.vec.print" (%v1, %ch0) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: vec = 0, 1, 2, 3, 4

  %v2 = "test.vec.constant"() { size = 5 : si32, value = 3 : si32 } : () -> !test.vec
  %ch2 = "test.vec.print" (%v2, %ch1) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: vec = 3, 3, 3, 3, 3

  %v3 = "test.vec.add.async2"(%v1, %v2) : (!test.vec, !test.vec) -> !test.vec
  %ch3 = "test.vec.print" (%v3, %ch2) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: vec = 3, 4, 5, 6, 7

  // CHECK-NEXT: 'async_add_test_par_foreach' returned chain

  mef.output %ch3 : !mef.chain
}

// CHECK-LABEL: --- Running 'async_accumulate'
grt.testcase @async_accumulate {
  %ch0 = grt.chain.create ()
  %v1 = "test.vec.iota"() { size = 100 : si32 } : () -> !test.vec

  %i1 = "test.vec.accumulate"(%v1) : (!test.vec) -> si32
  %ch1 = "grt.print.si32" (%i1, %ch0) : (si32, !mef.chain) -> !mef.chain
  // CHECK-NEXT: int32 = 4950

  %i2 = "test.vec.accumulate.async"(%v1) : (!test.vec) -> si32
  %ch2 = "grt.print.si32" (%i2, %ch1) : (si32, !mef.chain) -> !mef.chain
  // CHECK-NEXT: int32 = 4950

  %i3 = "test.vec.accumulate.async2"(%v1) : (!test.vec) -> si32
  %ch3 = "grt.print.si32" (%i3, %ch2) : (si32, !mef.chain) -> !mef.chain
  // CHECK-NEXT: int32 = 4950

  // CHECK-NEXT: 'async_accumulate' returned chain

  mef.output %ch3 : !mef.chain
}

// CHECK-LABEL: --- Running 'async_softmax'
grt.testcase @async_softmax {
  %ch0 = grt.chain.create ()
  %v1 = "test.vec.f32.iota"() { size = 10 : si32, step = 0.1 : f32 } : () -> !test.vec
  %v2 = "test.vec.f32.softmax.async"(%v1) : (!test.vec) -> !test.vec
  %ch1 = "test.vec.f32.print" (%v2, %ch0) : (!test.vec, !mef.chain) -> !mef.chain
  // CHECK-NEXT: vec.f32 = 0.061{{[0-9]+}}, 0.067{{[0-9]+}}, 0.074{{[0-9]+}}, 0.082{{[0-9]+}}, 0.091{{[0-9]+}}, 0.100{{[0-9]+}}, 0.111{{[0-9]+}}, 0.123{{[0-9]+}}, 0.136{{[0-9]+}}, 0.150{{[0-9]+}}

  %v3 = "test.vec.f32.constant"() { size = 100 : si32, value = 1.0 : f32 } : () -> !test.vec
  %v4 = "test.vec.f32.softmax.async"(%v3) : (!test.vec) -> !test.vec
  %ch2 = "test.vec.f32.print" (%v4, %ch1) : (!test.vec, !mef.chain) -> !mef.chain
  // The exact output is 0.01 repeated 100 times. We test that the output contains
  // exactly 100 elements all of which are 0.01.
  // CHECK-NEXT: vec.f32 = {{((0.01[0-9]+, ){99})}}0.01{{[0-9]+}}

  %v5 = "test.vec.f32.constant"() { size = 10000 : si32, value = 1.0 : f32 } : () -> !test.vec
  %v6 = "test.vec.f32.softmax.async"(%v5) : (!test.vec) -> !test.vec
  %ch3 = "test.vec.f32.print" (%v6, %ch2) : (!test.vec, !mef.chain) -> !mef.chain
  // The exact output is 0.0001 repeated 10000 times. We cannot test the output
  // contains exactly 10000 elements, as there is a 255 POSIX limit on the
  // repeated regular expression. So, we just test that the output starts with
  // 0.0001 and ends with 0.0001
  // CHECK-NEXT: vec.f32 = 0.0001{{[0-9]+}}, {{.*}}, 0.0001{{[0-9]+}}

  // CHECK-NEXT: 'async_softmax' returned chain

  mef.output %ch3 : !mef.chain
}
