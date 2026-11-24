// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor -verify-diagnostics %t.mef | FileCheck %s

// CHECK-LABEL: --- Running 'test_async_print_result'
grt.testcase @test_async_print_result {
  %c1 = grt.constant.si32 1
  %c42 = grt.constant.si32 42
  %a = "test.async.add.i32"(%c1, %c42) : (si32, si32) -> si32
  mef.output %a : si32
}
// CHECK: 'test_async_print_result' returned 43

// CHECK-LABEL: --- Running 'test_add_any'
grt.testcase @test_add_any {
  %c1 = grt.constant.si32 1
  %c42 = grt.constant.si32 42
  %a = "test.async.add.any"(%c1, %c42) : (si32, si32) -> si32

  %x = "test.get_string"() { value = "bark" } : () -> !test.string
  %y = "test.get_string"() { value = "rarf" } : () -> !test.string
  %b = "test.async.add.any"(%x, %y) : (!test.string, !test.string) -> !test.string

  // expected-error @+1 {{invalid operands to async_add_any}}
  %c = "test.async.add.any"(%c1, %y) : (si32, !test.string) -> si32

  mef.output %a, %b, %c : si32, !test.string, si32
}
// CHECK: 'test_add_any' returned 43, barkrarf, <error: invalid operands to async_add_any>


// CHECK-LABEL: --- Running 'test_async_errors'
grt.testcase @test_async_errors {
  // expected-error @+2 {{error generated on a background thread!}}
  // expected-error @+1 {{error generated on a background thread #2!}}
  %a, %b = "test.async.error"() : () -> (si32, !test.string)

  mef.output %a, %b : si32, !test.string
}
// CHECK: 'test_async_errors' returned
// CHECK: <error: error generated on a background thread!>,
// CHECK: <error: error generated on a background thread #2!>

// CHECK-LABEL: --- Running 'test_crash_on_threadpool_shutdown'
grt.testcase @test_crash_on_threadpool_shutdown {
  "test.crash.threadpool.shutdown"() : () -> ()
  mef.output
}
// CHECK-DAG: task 0 shutdown
// CHECK-DAG: task 1 shutdown
// CHECK-DAG: task 2 shutdown
// CHECK-DAG: task 3 shutdown
